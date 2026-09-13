import csv
from pathlib import Path
from scipy import ndimage
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, random_split


def create_boundary_target(
    instance_mask,
    boundary_width=2,
    adaptive=False,
):
    """
    Converte uma máscara de instâncias em um alvo de 3 classes:

        0 = background
        1 = interior
        2 = boundary

    A fronteira é construída dentro de cada instância por erosão
    morfológica.

    adaptive=True garante ao menos 1 px de interior por instância. Sem
    isso, um núcleo de 4 px perde o interior inteiro, o watershed fica
    sem marcador e o objeto some da saída (ver Parte 5).
    """

    target = np.zeros_like(instance_mask, dtype=np.int64)

    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0]

    for instance_id in instance_ids:

        mask = instance_mask == instance_id

        eroded = ndimage.binary_erosion(
            mask,
            iterations=boundary_width,
            border_value=0,
        )

        if adaptive and not eroded.any():

            # Afina a fronteira até sobrar interior.
            for iterations in range(boundary_width - 1, 0, -1):

                eroded = ndimage.binary_erosion(
                    mask,
                    iterations=iterations,
                    border_value=0,
                )

                if eroded.any():
                    break

            # Pequeno demais até para 1 px de fronteira: guarda o ponto
            # mais interno como interior.
            if not eroded.any():

                inner = ndimage.distance_transform_edt(mask)

                peak = np.unravel_index(
                    np.argmax(np.where(mask, inner, -1.0)),
                    inner.shape,
                )

                eroded = np.zeros_like(mask)
                eroded[peak] = True

        target[eroded] = 1
        target[mask & ~eroded] = 2

    return target


class DSB2018Dataset(Dataset):
    """
    Dataset loader for the DSB2018-style directory structure.

    Expected structure:

        root/
        ├── sample_1/
        │   ├── images/
        │   │   └── image.png
        │   └── masks/
        │       ├── instance_1.png
        │       ├── instance_2.png
        │       └── ...
        ├── sample_2/
        │   └── ...
        └── ...

    Each PNG inside masks/ represents one individual instance.
    """

    def __init__(self, root, transform=None, adaptive_boundary=False):
        self.root = Path(root)
        self.transform = transform
        # Ver src/postprocessing.boundary_prediction_to_instances_seeded
        # e scripts/failure_correction.py: com erosao fixa de 2 px, todo
        # nucleo com diametro <= 4 px fica sem interior no alvo e nunca
        # pode ser decodificado. adaptive_boundary=True garante ao menos
        # um pixel de interior por instancia.
        self.adaptive_boundary = adaptive_boundary

        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found in {self.root}"
            )

    def _find_samples(self):
        samples = []

        for sample_dir in sorted(self.root.iterdir()):

            if not sample_dir.is_dir():
                continue

            image_dir = sample_dir / "images"
            mask_dir = sample_dir / "masks"

            if not image_dir.exists() or not mask_dir.exists():
                continue

            images = sorted(image_dir.glob("*.png"))
            masks = sorted(mask_dir.glob("*.png"))

            if len(images) == 0:
                continue

            if len(masks) == 0:
                continue

            # DSB2018 has one image per sample.
            samples.append(
                {
                    "image": images[0],
                    "masks": masks,
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _load_instance_mask(self, mask_paths, shape):
        """
        Combine individual binary masks into one instance-label mask.

        Output:
            0 = background
            1 = first instance
            2 = second instance
            ...
        """

        instance_mask = np.zeros(shape, dtype=np.int32)

        for instance_id, mask_path in enumerate(mask_paths, start=1):

            mask = np.array(
                Image.open(mask_path).convert("L")
            )

            mask = mask > 0

            instance_mask[mask] = instance_id

        return instance_mask
    
    def _create_boundary_target(self, instance_mask, boundary_width=2):
        """
        Ver create_boundary_target, no nível do módulo.
        """

        return create_boundary_target(
            instance_mask,
            boundary_width=boundary_width,
            adaptive=self.adaptive_boundary,
        )

    def __getitem__(self, index):
        sample = self.samples[index]
    
        image = self._load_image(sample["image"])
    
        h, w = image.shape[:2]
    
        instance_mask = self._load_instance_mask(
            sample["masks"],
            (h, w)
        )
    
        # --------------------------------------------------------
        # Padroniza tamanho
        # --------------------------------------------------------
    
        target_size = (256, 256)
    
        image = Image.fromarray(image).resize(
            target_size,
            Image.Resampling.BILINEAR
        )
    
        instance_mask = Image.fromarray(instance_mask).resize(
            target_size,
            Image.Resampling.NEAREST
        )
        
    
        image = np.array(image)
        instance_mask = np.array(instance_mask)
        
        boundary_target = self._create_boundary_target(
        instance_mask,
        boundary_width=2
)
    
        # --------------------------------------------------------
        # Transformações opcionais
        # --------------------------------------------------------
    
        if self.transform is not None:
    
            transformed = self.transform(
                image=image,
                mask=instance_mask
            )
    
            image = transformed["image"]
            instance_mask = transformed["mask"]
    
        # --------------------------------------------------------
        # Tensor
        # --------------------------------------------------------
    
        image = torch.from_numpy(
            image.copy()
        ).float() / 255.0
    
        image = image.permute(2, 0, 1)
    
        instance_mask = torch.from_numpy(
            instance_mask.copy()
        ).long()
    
        return {
            "image": image,
            "instance_mask": instance_mask,
            "boundary_target": torch.from_numpy(
                boundary_target.copy()
            ).long(),
            "image_path": str(sample["image"])
        }
    
# ----------------------------------------------------------------------
# Split estratificado por modalidade
# ----------------------------------------------------------------------
#
# O enunciado exige "split treino/validacao/teste estratificado por
# modalidade ou cidade, justificado na apresentacao". O DSB2018 nao traz
# rotulo de modalidade, entao derivamos um a partir de duas estatisticas
# da propria imagem:
#
#   separacao de canais = media de (max - min) entre os canais RGB
#   brilho              = media dos canais
#
# Os cortes caem em vazios grandes do histograma, e nao no meio de nada:
# a separacao de canais e EXATAMENTE 0.0 em 562 das 670 imagens e >= 30
# nas outras 108; entre as acinzentadas, o brilho e <= 57 (fluorescencia)
# ou >= 205 (campo claro), sem nada no meio. Um KMeans(k=4) independente
# sobre as duas features reencontra os mesmos tres grupos.

CHANNEL_SEPARATION_THRESHOLD = 5.0
BRIGHTNESS_THRESHOLD = 100.0

MODALITIES = ("fluor_escura", "brightfield_clara", "histologia_colorida")

MODALITY_CACHE = Path("experiments/results/modalidades.csv")


def image_statistics(image_path):
    """
    (brilho medio, separacao media de canais) de uma imagem.
    """

    image = np.array(
        Image.open(image_path).convert("RGB"),
        dtype=np.float32,
    )

    brightness = float(image.mean())

    separation = float(
        (image.max(axis=2) - image.min(axis=2)).mean()
    )

    return brightness, separation


def classify_modality(brightness, separation):
    """
    Rotulo de modalidade a partir das duas estatisticas.
    """

    if separation >= CHANNEL_SEPARATION_THRESHOLD:
        return "histologia_colorida"

    if brightness < BRIGHTNESS_THRESHOLD:
        return "fluor_escura"

    return "brightfield_clara"


def dataset_modalities(dataset, cache_path=MODALITY_CACHE):
    """
    Modalidade de cada amostra do dataset, na ordem dos indices.

    Le as 670 imagens uma vez e guarda em CSV; as chamadas seguintes
    (todo script chama create_splits) saem do cache.
    """

    paths = [str(sample["image"]) for sample in dataset.samples]

    if cache_path is not None and Path(cache_path).exists():

        cached = {}

        with open(cache_path) as file:
            for row in csv.DictReader(file):
                cached[row["image_path"]] = row["modality"]

        if all(path in cached for path in paths):
            return [cached[path] for path in paths]

    modalities = []
    rows = []

    for path in paths:

        brightness, separation = image_statistics(path)

        modality = classify_modality(brightness, separation)

        modalities.append(modality)

        rows.append({
            "image_path": path,
            "brightness": round(brightness, 2),
            "channel_separation": round(separation, 2),
            "modality": modality,
        })

    if cache_path is not None:

        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image_path", "brightness",
                    "channel_separation", "modality",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

    return modalities


def create_splits(
    dataset,
    seed=42,
    train_ratio=0.8,
    val_ratio=0.1,
    stratify=True,
):
    """
    Divide o dataset em treino / validacao / teste.

    stratify=True (padrao) preserva a proporcao de cada modalidade nos
    tres splits, como o enunciado exige. stratify=False reproduz o
    random_split antigo, para comparacao.

    Devolve tres Subset, iguais ao que random_split devolvia.
    """

    n_total = len(dataset)

    if not stratify:

        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        generator = torch.Generator().manual_seed(seed)

        return random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=generator,
        )

    modalities = dataset_modalities(dataset)

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    for modality in sorted(set(modalities)):

        indices = np.array(
            [i for i, m in enumerate(modalities) if m == modality]
        )

        rng.shuffle(indices)

        n_group = len(indices)

        n_train = int(round(train_ratio * n_group))
        n_val = int(round(val_ratio * n_group))

        # Grupos pequenos (brightfield_clara tem 16 imagens) nao podem
        # ficar sem representacao em validacao e teste — e exatamente o
        # que a estratificacao existe para evitar.
        if n_group >= 3:
            n_val = max(n_val, 1)
            n_train = min(n_train, n_group - 2)

        n_train = max(n_train, 0)

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val:].tolist())

    # Ordena para o conjunto de teste ter ordem estavel entre execucoes.
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def split_report(dataset, splits, printer=print):
    """
    Tabela de distribuicao de modalidade por split — a justificativa que
    a apresentacao precisa mostrar.
    """

    modalities = dataset_modalities(dataset)

    names = ("treino", "validacao", "teste")

    printer(f"{'modalidade':<22}{'treino':>9}{'validacao':>11}{'teste':>8}{'total':>8}")

    totals = {name: 0 for name in names}

    for modality in MODALITIES:

        counts = []

        for split in splits:
            counts.append(
                sum(1 for i in split.indices if modalities[i] == modality)
            )

        if sum(counts) == 0:
            continue

        for name, count in zip(names, counts):
            totals[name] += count

        printer(
            f"{modality:<22}{counts[0]:>9}{counts[1]:>11}{counts[2]:>8}"
            f"{sum(counts):>8}"
        )

    printer(
        f"{'TOTAL':<22}{totals['treino']:>9}{totals['validacao']:>11}"
        f"{totals['teste']:>8}{sum(totals.values()):>8}"
    )

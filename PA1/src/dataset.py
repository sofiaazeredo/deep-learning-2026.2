from pathlib import Path
from scipy import ndimage
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


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

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

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
        Converte uma máscara de instâncias em um target de 3 classes:
    
            0 = background
            1 = interior
            2 = boundary
    
        A boundary é construída dentro de cada instância através
        de erosão morfológica.
        """
    
        target = np.zeros_like(
            instance_mask,
            dtype=np.int64
        )
    
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids != 0]
    
        for instance_id in instance_ids:
    
            mask = instance_mask == instance_id
    
            # Erode o objeto
            eroded = ndimage.binary_erosion(
                mask,
                iterations=boundary_width,
                border_value=0
            )
    
            # Pixels removidos pela erosão = boundary
            boundary = mask & ~eroded
    
            # Interior
            target[eroded] = 1
    
            # Boundary
            target[boundary] = 2
    
        return target

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
    
def create_splits(dataset, seed=42, train_ratio=0.8, val_ratio=0.1):
    from torch.utils.data import random_split

    n_total = len(dataset)

    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)

    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

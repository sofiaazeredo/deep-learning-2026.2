from pathlib import Path

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
            "image_path": str(sample["image"])
        }

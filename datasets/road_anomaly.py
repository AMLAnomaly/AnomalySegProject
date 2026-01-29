from pathlib import Path
from typing import Callable, Optional, List, Tuple
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
import numpy as np
import os
import glob

class RoadAnomaly(TorchDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.targets = []


        
        search_pattern = os.path.join(root, "images", "*.jpg")
        image_paths = glob.glob(search_pattern)
        
        for path in image_paths:
             self.images.append(path)

             target_path = path.replace("images", "labels_masks").replace(".jpg", ".png")
             self.targets.append(target_path)
             
        if len(self.images) == 0:
             print(f"WARNING: RoadAnomaly dataset empty at {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.Tensor]:
        img = Image.open(self.images[index]).convert("RGB")
        target_img = Image.open(self.targets[index])
        

        

        # - 2 means anomaly so it's mapped to 1
        # - other label values remain the same
        target_np = np.array(target_img, dtype=np.uint8)
        target_np = np.where(target_np == 2, 1, target_np).astype(np.uint8)
        target = torch.from_numpy(target_np).long()

        if self.transform is not None:
             img = self.transform(img)


        return img, target

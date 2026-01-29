# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import re
import json
import zipfile
import os
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional
from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class FolderProxy:
    """Mimics zipfile.ZipFile for unzipped directories."""
    def __init__(self, root: Path):
        self.root = Path(root)

    def open(self, name, mode="r"):

        clean_name = str(name).lstrip("./").lstrip("/")
        return open(self.root / clean_name, "rb")

    def namelist(self):
        files = [
            p.relative_to(self.root).as_posix() 
            for p in self.root.rglob("*") 
            if p.is_file()
        ]
        return files

    def infolist(self):
        return [zipfile.ZipInfo(filename=f) for f in self.namelist()]

    def close(self):
        pass
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zip_path: Path,
        img_suffix: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
        only_annotations_json: bool = False,
        target_suffix: str = None,
        stuff_classes: Optional[list[int]] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_zip_path: Optional[Path] = None,
        target_zip_path_in_zip: Optional[Path] = None,
        target_instance_zip_path: Optional[Path] = None,
        img_folder_path_in_zip: Path = Path("./"),
        target_folder_path_in_zip: Path = Path("./"),
        target_instance_folder_path_in_zip: Path = Path("./"),
        annotations_json_path_in_zip: Optional[Path] = None,
    ):
        self.zip_path = Path(zip_path) 
        self.target_parser = target_parser
        self.transforms = transforms
        self.only_annotations_json = only_annotations_json
        self.stuff_classes = stuff_classes
        self.target_zip_path = target_zip_path
        self.target_zip_path_in_zip = target_zip_path_in_zip
        self.target_instance_zip_path = target_instance_zip_path
        self.target_folder_path_in_zip = target_folder_path_in_zip
        self.target_instance_folder_path_in_zip = target_instance_folder_path_in_zip

        self.zip = None
        self.target_zip = None
        self.target_instance_zip = None
        img_zip, target_zip, target_instance_zip = self._load_zips()

        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}

        if annotations_json_path_in_zip is not None:
            with zipfile.ZipFile(target_zip_path or zip_path) as outer_target_zip:
                with outer_target_zip.open(
                    str(annotations_json_path_in_zip), "r"
                ) as file:
                    annotation_data = json.load(file)

            image_id_to_file_name = {
                image["id"]: image["file_name"] for image in annotation_data["images"]
            }

            for annotation in annotation_data["annotations"]:
                img_filename = image_id_to_file_name[annotation["image_id"]]

                if "segments_info" in annotation:
                    self.labels_by_id[img_filename] = {
                        segment_info["id"]: segment_info["category_id"]
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_crowd_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info["iscrowd"])
                        for segment_info in annotation["segments_info"]
                    }
                else:
                    if img_filename not in self.labels_by_id:
                        self.labels_by_id[img_filename] = {}

                    if img_filename not in self.polygons_by_id:
                        self.polygons_by_id[img_filename] = {}

                    if img_filename not in self.is_crowd_by_id:
                        self.is_crowd_by_id[img_filename] = {}

                    self.labels_by_id[img_filename][annotation["id"]] = annotation[
                        "category_id"
                    ]
                    self.polygons_by_id[img_filename][annotation["id"]] = annotation[
                        "segmentation"
                    ]
                    self.is_crowd_by_id[img_filename][annotation["id"]] = bool(
                        annotation["iscrowd"]
                    )

        self.imgs = []
        self.targets = []
        self.targets_instance = []

        target_zip_filenames = target_zip.namelist()
        input_files = sorted(img_zip.infolist(), key=self._sort_key)
        
        
        count_valid = 0
        debug_rejected_count = 0
        left_img_found = False
        
        for img_info in input_files:
            if "leftImg8bit" in img_info.filename and not left_img_found:
                 print(f"[DEBUG] Found a leftImg8bit file: {img_info.filename}")
                 left_img_found = True
                 
            if not self.valid_member(
                img_info, img_folder_path_in_zip, img_stem_suffix, img_suffix
            ):
                if debug_rejected_count < 5:
                     pass
                continue
            
            if count_valid == 0:
                print(f"[DEBUG] SUCCESS! Accepted first file: {img_info.filename}")
                
            count_valid += 1
            img_path = Path(img_info.filename)
            if not only_annotations_json:
                rel_path = img_path.relative_to(img_folder_path_in_zip)
                target_parent = target_folder_path_in_zip / rel_path.parent
                target_stem = rel_path.stem.replace(img_stem_suffix, target_stem_suffix)

                target_filename = (target_parent / f"{target_stem}{target_suffix}").as_posix()

            if self.labels_by_id:
                if img_path.name not in self.labels_by_id:
                    continue

                if not self.labels_by_id[img_path.name]:
                    continue
            else:
                if target_filename not in target_zip_filenames:
                    continue

                if check_empty_targets:
                    with target_zip.open(target_filename) as target_file:
                        min_val, max_val = Image.open(target_file).getextrema()
                        if min_val == max_val:
                            continue

            if target_instance_zip is not None:
                target_instance_filename = (
                    target_instance_folder_path_in_zip / (target_stem + target_suffix)
                ).as_posix()

                if check_empty_targets:
                    with target_instance_zip.open(
                        target_instance_filename
                    ) as target_instance:
                        extrema = Image.open(target_instance).getextrema()
                        if all(min_val == max_val for min_val, max_val in extrema):
                            _, labels, _ = self.target_parser(
                                target=tv_tensors.Mask(
                                    Image.open(target_zip.open(target_filename))
                                ),
                                target_instance=tv_tensors.Mask(
                                    Image.open(target_instance)
                                ),
                                stuff_classes=self.stuff_classes,
                            )
                            if not labels:
                                continue

            self.imgs.append(img_path.as_posix())

            if not only_annotations_json:
                self.targets.append(target_filename)

            if target_instance_zip is not None:
                self.targets_instance.append(target_instance_filename)

    def __getitem__(self, index: int):
        img_zip, target_zip, target_instance_zip = self._load_zips()

        with img_zip.open(self.imgs[index]) as img:
            img = tv_tensors.Image(Image.open(img).convert("RGB"))

        target = None
        if not self.only_annotations_json:
            with target_zip.open(self.targets[index]) as target_file:
                target = tv_tensors.Mask(Image.open(target_file), dtype=torch.long)

            if img.shape[-2:] != target.shape[-2:]:
                target = F.resize(
                    target,
                    list(img.shape[-2:]),
                    interpolation=F.InterpolationMode.NEAREST,
                )

        target_instance = None
        if self.targets_instance:
            with target_instance_zip.open(
                self.targets_instance[index]
            ) as target_instance:
                target_instance = tv_tensors.Mask(
                    Image.open(target_instance), dtype=torch.long
                )

        masks, labels, is_crowd = self.target_parser(
            target=target,
            target_instance=target_instance,
            stuff_classes=self.stuff_classes,
            polygons_by_id=self.polygons_by_id.get(Path(self.imgs[index]).name, {}),
            labels_by_id=self.labels_by_id.get(Path(self.imgs[index]).name, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(Path(self.imgs[index]).name, {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _load_zips(
        self,
    ) -> Tuple[object, object, Optional[object]]:
        worker = get_worker_info()
        worker = worker.id if worker else None

        if self.zip is None:
            self.zip = {}
        if self.target_zip is None:
            self.target_zip = {}
        if self.target_instance_zip is None and self.target_instance_zip_path:
            self.target_instance_zip = {}

        def get_loader(path: Path):
            if path is None:
                return None
            path = Path(path)
            if path.suffix == ".zip":
                return zipfile.ZipFile(path)
            elif path.is_dir():
                return FolderProxy(path)
            else:
                 if path.is_dir():
                     return FolderProxy(path)
                 raise ValueError(f"Path {path} is neither a zip file nor a directory.")

        if worker not in self.zip:
            self.zip[worker] = get_loader(self.zip_path)
            
        if worker not in self.target_zip:
            if self.target_zip_path:
                self.target_zip[worker] = get_loader(self.target_zip_path)
                

                if self.target_zip_path_in_zip and isinstance(self.target_zip[worker], zipfile.ZipFile):
                    with self.target_zip[worker].open(
                        str(self.target_zip_path_in_zip)
                    ) as target_zip_stream:
                        nested_zip_data = BytesIO(target_zip_stream.read())
                    self.target_zip[worker].close()
                    self.target_zip[worker] = zipfile.ZipFile(nested_zip_data)
            else:
                self.target_zip[worker] = self.zip[worker]
                
        if (
            self.target_instance_zip_path is not None
            and worker not in self.target_instance_zip
        ):
            self.target_instance_zip[worker] = get_loader(self.target_instance_zip_path)

        return (
            self.zip[worker],
            self.target_zip[worker],
            self.target_instance_zip[worker] if self.target_instance_zip_path else None,
        )

    @staticmethod
    def _sort_key(m: zipfile.ZipInfo):
        match = re.search(r"\d+", m.filename)

        return (int(match.group()) if match else float("inf"), m.filename)

    @staticmethod
    def valid_member(
        img_info: zipfile.ZipInfo,
        img_folder_path_in_zip: Path,
        img_stem_suffix: str,
        img_suffix: str,
    ):
        fname = Path(img_info.filename).as_posix()
        folder = img_folder_path_in_zip.as_posix()
        
        if not fname.startswith(folder):
            return False
            
        expected_suffix = img_stem_suffix + img_suffix
        if not fname.endswith(expected_suffix):
            return False
            
        if img_info.is_dir():
            return False
            
        return True

    def __len__(self):
        return len(self.imgs)

    def close(self):
        if self.zip is not None:
            for item in self.zip.values():
                item.close()
            self.zip = None

        if self.target_zip is not None:
            for item in self.target_zip.values():
                item.close()
            self.target_zip = None

        if self.target_instance_zip is not None:
            for item in self.target_instance_zip.values():
                item.close()
            self.target_instance_zip = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["zip"] = None
        state["target_zip"] = None
        state["target_instance_zip"] = None
        return state

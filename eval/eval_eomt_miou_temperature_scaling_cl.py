

import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import Compose


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
eomt_root = os.path.join(project_root, "eomt")

for p in [project_root, eomt_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from eomt_utils import build_eomt_cityscapes_lit_model_for_eval
from dataset import cityscapes
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry


NUM_CLASSES = 20  
CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "Traffic light", "Traffic sign", "Vegetation", "Terrain",
    "Sky", "Person", "Rider", "Car", "Truck",
    "Bus", "Train", "Motorcycle", "Bicycle"
]

ID_TO_TRAINID = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
}


class CityscapesIdToTrainId(object):
    #Converts Cityscapes labelIds to trainIds.

    def __call__(self, label_img):
        label = np.array(label_img, dtype=np.int64)
        trainId = np.full_like(label, 255, dtype=np.int64)

        for id_val, train_id in ID_TO_TRAINID.items():
            trainId[label == id_val] = train_id

        return Image.fromarray(trainId.astype(np.uint8))


def pil_to_uint8(pic):
    return torch.from_numpy(np.array(pic)).permute(2, 0, 1)


input_transform_cityscapes = Compose([
    pil_to_uint8, 
])

target_transform_cityscapes = Compose([
    CityscapesIdToTrainId(), 
    ToLabel(),                
    Relabel(255, 19),      
])



def main(args):
    device = torch.device("cpu" if args.cpu else "cuda")
    temperatures = [0.5, 0.75, 1.0, 1.1]

    weightspath = os.path.join(args.loadDir, args.loadWeights)
    print("Loading EoMT checkpoint:", weightspath)
    model = build_eomt_cityscapes_lit_model_for_eval(weightspath, device, cpu=args.cpu)
    print("EoMT model loaded successfully.")

    if not os.path.exists(args.datadir):
        raise FileNotFoundError(f"Dataset directory not found: {args.datadir}")

    ds = cityscapes(
        args.datadir,
        input_transform_cityscapes,
        target_transform_cityscapes,
        subset=args.subset
    )

    loader = DataLoader(
        ds,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    iou_evaluators = {t: iouEval(NUM_CLASSES) for t in temperatures}
    start = time.time()

    for step, (images, labels, filename, _) in enumerate(loader):
        if not args.cpu:
            images = images.to(device)
            labels = labels.to(device)

        img_sizes = [img.shape[-2:] for img in images]

        with torch.no_grad():
            crops, origins = model.window_imgs_semantic(images)
            mask_logits_per_layer, class_logits_per_layer = model(crops)

            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], model.img_size, mode="bilinear", align_corners=False
            )
            base_class_logits = class_logits_per_layer[-1]

            for temp in temperatures:
                scaled_class_logits = base_class_logits / temp
                crop_logits = model.to_per_pixel_logits_semantic(mask_logits, scaled_class_logits)
                pixel_logits_list = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
                pixel_logits = torch.stack(pixel_logits_list)
                preds = pixel_logits.argmax(dim=1, keepdim=True)
                iou_evaluators[temp].addBatch(preds.data, labels)

        short_name = filename[0].split("leftImg8bit/")[-1] if "leftImg8bit/" in filename[0] else os.path.basename(filename[0])
        print(step, short_name)

    elapsed = time.time() - start
    print("Processing time (s):", elapsed)

    if not os.path.exists('results'):
        os.makedirs('results')

    results_path = os.path.join("results", "results_EoMT_Temperature_mIoU_cl.txt")
    with open(results_path, "a") as result_file:
        result_file.write(f"Subset: {args.subset}\n")

        for temp in temperatures:
            iou_val, iou_classes = iou_evaluators[temp].getIoU()
            print(f"Temperature {temp}")
            for idx in range(iou_classes.size(0)):
                entry = getColorEntry(iou_classes[idx]) + f"{iou_classes[idx] * 100:.2f}" + '\033[0m'
                print(entry, CLASS_NAMES[idx])
            mean_iou_str = getColorEntry(iou_val) + f"{iou_val * 100:.2f}" + '\033[0m'
            print(f"Mean IoU: {mean_iou_str}%")

            result_file.write(f"Temperature {temp}: Mean IoU = {iou_val * 100:.2f}%\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="epoch_106-step-19902_eomt.ckpt")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.5, 0.75, 1.0, 1.1])

    main(parser.parse_args())

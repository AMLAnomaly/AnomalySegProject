

import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
eomt_root = os.path.join(project_root, "eomt")

for p in [project_root, eomt_root]:
    if p not in sys.path:
        sys.path.insert(0, p)



from eomt_utils import (
    build_eomt_cityscapes_lit_model_for_eval
)

from dataset import cityscapes
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry


NUM_CLASSES = 20
IMG_SIZE = (1024, 1024) 


ID_TO_TRAINID = {
    0: 255,   # unlabeled
    1: 255,   # ego vehicle
    2: 255,   # rectification border
    3: 255,   # out of roi
    4: 255,   # static
    5: 255,   # dynamic
    6: 255,   # ground
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
}


class CityscapesIdToTrainId(object):
    #Converts from labelIds (0..33, 255) to trainIds (0..18, 255)

    def __call__(self, label_img):
        label = np.array(label_img, dtype=np.int64)
        trainId = np.full_like(label, 255, dtype=np.int64)

        for id_val, train_id in ID_TO_TRAINID.items():
            trainId[label == id_val] = train_id

        return Image.fromarray(trainId.astype(np.uint8))


def pil_to_uint8(pic):

    return torch.from_numpy(np.array(pic)).permute(2, 0, 1)


input_transform_cityscapes = Compose([
    pil_to_uint8,  # -> [0,1]
])

target_transform_cityscapes = Compose([
    CityscapesIdToTrainId(),  
    ToLabel(),                
    Relabel(255, 19),        
])




def main(args):
    device = torch.device("cpu" if args.cpu else "cuda")

    weightspath = os.path.join(args.loadDir, args.loadWeights)
    print("Loading EoMT checkpoint:", weightspath)

    model = build_eomt_cityscapes_lit_model_for_eval(weightspath, device, cpu=args.cpu)
    print("EoMT model loaded successfully.")

    if not os.path.exists(args.datadir):
        print("Error: datadir could not be loaded:", args.datadir)
        return

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

    print("Dataset size (len(loader.dataset)) =", len(loader.dataset))

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):

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

            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits, class_logits_per_layer[-1]
            )

            pixel_logits_list = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            
            pixel_logits = torch.stack(pixel_logits_list)

            preds = pixel_logits.argmax(dim=1, keepdim=True)

        iouEvalVal.addBatch(preds.data, labels)

        if "leftImg8bit/" in filename[0]:
            filenameSave = filename[0].split("leftImg8bit/")[1]
        else:
            filenameSave = os.path.basename(filename[0])

        print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = (
            getColorEntry(iou_classes[i]) +
            '{:0.2f}'.format(iou_classes[i] * 100) +
            '\033[0m'
        )
        iou_classes_str.append(iouStr)

    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
    print("MEAN IoU: ", iouStr, "%")

    if not os.path.exists('results'):
        os.makedirs('results')

    with open(os.path.join("results", "results_EoMT.txt"), "a") as f:
        f.write(f"MEAN IoU: {iouVal * 100:.2f} %\n")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="epoch_106-step-19902_eomt.ckpt")
    parser.add_argument('--subset', default="val")  # 'val' or 'train'
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())

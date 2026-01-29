

import numpy as np
import torch
import torch.nn.functional as F
import os
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20  # 19 classes + 1 ignore (19)

image_transform = ToPILImage()


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
        # default: ignore (255)
        trainId = np.full_like(label, 255, dtype=np.int64)

        for id_val, train_id in ID_TO_TRAINID.items():
            trainId[label == id_val] = train_id

        return Image.fromarray(trainId.astype(np.uint8))



input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])

target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    CityscapesIdToTrainId(),   # labelIds -> trainIds
    ToLabel(),
    Relabel(255, 19),          # ignore label 255 -> 19
])


def main(args):

    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        """Custom loader to handle potential 'module.' in names."""
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    key = name.split("module.")[-1]
                    if key in own_state:
                        own_state[key].copy_(param)
                    else:
                        print(name, " not loaded")
                else:
                    print(name, " not loaded")
                continue
            else:
                own_state[name].copy_(param)
        return model

    state_dict = torch.load(
        weightspath,
        map_location=lambda storage, loc: storage
    )
    model = load_my_state_dict(model, state_dict)
    print("Model and weights LOADED successfully")

    model.eval()

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

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if not args.cpu:
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)  

        # Per-pixel prediction: argmax on class channels
        preds = outputs.max(1)[1].unsqueeze(1).data  

        iouEvalVal.addBatch(preds, labels)

        if "leftImg8bit/" in filename[0]:
            filenameSave = filename[0].split("leftImg8bit/")[1]
        else:
            filenameSave = os.path.basename(filename[0])

        print(step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = (
            getColorEntry(iou_classes[i])
            + '{:0.2f}'.format(iou_classes[i] * 100)
            + '\033[0m'
        )
        iou_classes_str.append(iouStr)

    print("Took ", time.time() - start, "seconds")
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
    iouStr = (
        getColorEntry(iouVal)
        + '{:0.2f}'.format(iouVal * 100)
        + '\033[0m'
    )
    print("MEAN IoU: ", iouStr, "%")
    
    if not os.path.exists('results'):
        os.makedirs('results')

    with open(os.path.join("results", "results_ERFnet.txt"), "a") as f:
        f.write(f"MEAN IoU: {iouVal * 100:.2f} %\n")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # 'val' o 'train'
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())

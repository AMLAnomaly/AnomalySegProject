import torch
import torch.nn as nn
import torch.nn.functional as F


from models.eomt import EoMT
from models.vit import ViT
from training.mask_classification_semantic import MaskClassificationSemantic


def build_eomt_cityscapes_lit_model_for_eval(weightspath: str, device: torch.device, cpu: bool = False):

    encoder = ViT(
        backbone_name="vit_base_patch14_reg4_dinov2",
        img_size=(1024, 1024),
    )

    network = EoMT(
        encoder=encoder,
        num_classes=19,
        num_q=100,
        num_blocks=3,
        masked_attn_enabled=False,
    )


    lit_model = MaskClassificationSemantic(
        network=network,
        img_size=(1024, 1024), 
        num_classes=19,
        attn_mask_annealing_enabled=False,
        attn_mask_annealing_start_steps=[3317, 8292, 13268],
        attn_mask_annealing_end_steps=[6634, 11609, 16585],
        ckpt_path=weightspath, 
    )

    lit_model = lit_model.to(device)
    lit_model.eval()

    return lit_model



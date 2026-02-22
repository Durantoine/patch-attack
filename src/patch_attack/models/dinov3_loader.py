from pathlib import Path

import torch

_REPO = Path(__file__).parent / "dinov3"
_WEIGHTS = Path(__file__).parent / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def load_dinov3(device=None):
    """Charge DINOv3 ViT-S/16 depuis le dépôt local, poids gelés."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model = torch.hub.load(str(_REPO), "dinov3_vits16", source="local", pretrained=False)
    if _WEIGHTS.exists():
        model.load_state_dict(
            torch.load(_WEIGHTS, map_location=device, weights_only=False), strict=True
        )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

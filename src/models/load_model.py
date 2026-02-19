import torch

MODEL_NAME = "dinov3_vits16"
DINOV3_LOCATION = "."

dinov3_model = torch.hub.load(
    DINOV3_LOCATION,
    MODEL_NAME,
    source="local",
    weights="../dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
)

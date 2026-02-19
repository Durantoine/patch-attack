from pathlib import Path

import torch


def load_dinov3(checkpoint_path=None, model_name="dinov3_vits16", device="cuda"):
    """Load DINOv3 model from local repository.

    Args:
        checkpoint_path: Path to .pth checkpoint file (optional)
        model_name: Model architecture name
        device: Device to load model on

    Returns:
        DINOv3 model ready for inference
    """
    repo_path = Path(__file__).parent / "dinov3"

    if checkpoint_path is None:
        checkpoint_path = repo_path / f"{model_name}.pth"

    model = torch.hub.load(str(repo_path), model_name, source="local", pretrained=False)

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=True)

    model = model.to(device)
    model.eval()

    return model

"""Encoder implementations for UNet."""

from .vgg import VGGEncoder
from .resnet import ResNetEncoder
from .vit import ViTEncoder

ENCODER_REGISTRY = {
    "vgg": VGGEncoder,
    "resnet": ResNetEncoder,
    "vit": ViTEncoder,
}


def build_encoder(name: str, **kwargs):
    """Build an encoder by name."""
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODER_REGISTRY.keys())}")
    return ENCODER_REGISTRY[name](**kwargs)


__all__ = ["VGGEncoder", "ResNetEncoder", "ViTEncoder", "build_encoder", "ENCODER_REGISTRY"]

"""Decoder implementations for UNet."""

from .vgg import VGGDecoder
from .resnet import ResNetDecoder
from .vit import ViTDecoder

DECODER_REGISTRY = {
    "vgg": VGGDecoder,
    "resnet": ResNetDecoder,
    "vit": ViTDecoder,
}


def build_decoder(name: str, **kwargs):
    """Build a decoder by name."""
    if name not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder: {name}. Available: {list(DECODER_REGISTRY.keys())}")
    return DECODER_REGISTRY[name](**kwargs)


__all__ = ["VGGDecoder", "ResNetDecoder", "ViTDecoder", "build_decoder", "DECODER_REGISTRY"]

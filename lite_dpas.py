"""Compatibility imports for the Lite-DPAS model definitions."""

from src.models.lite_dpas import (
    LiteDPASCsiNet,
    LiteDPASCsiNetV2,
    LiteDPASCsiNetV3,
    LiteDPASCsiNetV4,
    LiteDPASCsiNetV5,
    count_parameters,
)

__all__ = [
    "LiteDPASCsiNet",
    "LiteDPASCsiNetV2",
    "LiteDPASCsiNetV3",
    "LiteDPASCsiNetV4",
    "LiteDPASCsiNetV5",
    "count_parameters",
]

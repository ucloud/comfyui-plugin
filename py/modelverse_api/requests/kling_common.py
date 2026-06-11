"""Shared helpers for Kling V3 request builders."""
import base64
from typing import Optional

from torch import Tensor

from ..utils import encode_image, tensor2images


MODEL_KLING_V3 = "kling-v3"
MODEL_KLING_V3_OMNI = "kling-v3-omni"

MODES = ["std", "pro"]
SOUNDS = ["off", "on"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
KLING_V3_TYPES = ["auto", "t2v", "i2v", "motion_control"]
CHARACTER_ORIENTATIONS = ["image", "video"]
YES_NO = ["yes", "no"]
REFER_TYPES = ["feature", "base"]


def normalize_kling_image_value(value: str) -> str:
    """Kling accepts http(s) URLs or raw base64 without a data:image prefix."""
    value = value.strip()
    if value.startswith("data:image/") and "," in value:
        return value.split(",", 1)[1]
    return value


def resolve_image(image: Optional[Tensor], url: str, label: str) -> Optional[str]:
    has_url = url and url.strip()
    has_image = image is not None
    if has_url and has_image:
        raise ValueError(f"{label}: provide either image or url, not both")
    if has_url:
        return normalize_kling_image_value(url)
    if has_image:
        data_bytes, _ = encode_image(tensor2images(image)[0])
        if not data_bytes:
            raise ValueError(f"{label}: failed to convert image to base64")
        return base64.b64encode(data_bytes).decode("utf-8")
    return None


def normalize_prompt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        return ""
    return str(value).strip()

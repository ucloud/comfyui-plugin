from typing import Optional
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor


def _image_to_raw_base64(image: Tensor) -> Optional[str]:
    """Encode an image tensor to a plain base64 string (no data URI prefix)."""
    data_url = image_to_base64(image)
    if not data_url:
        return None
    return data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url


class FluxPro11(BaseRequest):
    """
    Flux Pro 1.1 image generation.

    Endpoint: POST /v1/flux-pro-1.1
    """
    API_PATH = "/v1/flux-pro-1.1"
    prompt: str = Field(..., description="The prompt to generate an image from.")
    image_prompt: Optional[Tensor] = Field(
        default=None, description="Optional base64 image prompt.")
    width: Optional[int] = Field(
        default=1024, description="Image width in pixels, 256..1440 and a multiple of 32.")
    height: Optional[int] = Field(
        default=1024, description="Image height in pixels, 256..1440 and a multiple of 32.")
    prompt_upsampling: Optional[bool] = Field(
        default=False, description="Whether to upsample the prompt.")
    seed: Optional[int] = Field(
        default=-1, description="Optional seed for reproducible generation.")
    safety_tolerance: Optional[int] = Field(
        default=2, description="Moderation tolerance level, 0 (strictest) to 6 (loosest).")
    output_format: Optional[str] = Field(
        default="png", description="Output format: png, jpg or jpeg.")

    def __init__(
            self,
            prompt: str,
            image_prompt: Optional[Tensor] = None,
            width: Optional[int] = 1024,
            height: Optional[int] = 1024,
            prompt_upsampling: Optional[bool] = False,
            seed: Optional[int] = -1,
            safety_tolerance: Optional[int] = 2,
            output_format: Optional[str] = "png",
            **kwargs):
        super().__init__(**kwargs)

        self.prompt = prompt
        self.image_prompt = _image_to_raw_base64(
            image_prompt) if isinstance(image_prompt, Tensor) else None
        self.width = width
        self.height = height
        self.prompt_upsampling = prompt_upsampling
        self.seed = seed
        self.safety_tolerance = safety_tolerance
        self.output_format = output_format

    def build_payload(self) -> dict:
        payload = {
            "prompt": self.prompt,
            "image_prompt": self.image_prompt,
            "width": self.width,
            "height": self.height,
            "prompt_upsampling": self.prompt_upsampling,
            "safety_tolerance": self.safety_tolerance,
            "output_format": self.output_format,
        }
        if self.seed is not None and self.seed >= 0:
            payload["seed"] = self.seed
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return ["prompt", "image_prompt", "width", "height", "prompt_upsampling",
                "seed", "safety_tolerance", "output_format"]

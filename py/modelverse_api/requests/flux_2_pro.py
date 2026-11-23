from typing import Optional, List
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor


def _image_to_raw_base64(image: Tensor) -> Optional[str]:
    """Encode an image tensor to a plain base64 string (no data URI prefix)."""
    data_url = image_to_base64(image)
    if not data_url:
        return None
    return data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url


class Flux2Pro(BaseRequest):
    """
    Flux 2 Pro image generation / reference-image editing.

    Endpoint: POST /v1/flux-2-pro
    """
    API_PATH = "/v1/flux-2-pro"
    prompt: str = Field(..., description="The prompt to generate an image from.")
    input_images: Optional[list[Tensor]] = Field(
        default=None, description="Reference images, at most 8.")
    width: Optional[int] = Field(
        default=1024, description="Image width in pixels, >= 64 and a multiple of 32.")
    height: Optional[int] = Field(
        default=1024, description="Image height in pixels, >= 64 and a multiple of 32.")
    seed: Optional[int] = Field(
        default=-1, description="Optional seed for reproducible generation.")
    safety_tolerance: Optional[str] = Field(
        default="2", description="Moderation tolerance level, 0 (strictest) to 6 (loosest).")
    output_format: Optional[str] = Field(
        default="png", description="Output format: png, jpg or jpeg.")

    def __init__(
            self,
            prompt: str,
            input_images: Optional[List[Tensor]] = None,
            width: Optional[int] = 1024,
            height: Optional[int] = 1024,
            seed: Optional[int] = -1,
            safety_tolerance: Optional[str] = "2",
            output_format: Optional[str] = "png",
            **kwargs):
        super().__init__(**kwargs)

        self.prompt = prompt
        self.input_images = [_image_to_raw_base64(image)
                             for image in input_images] if input_images else None
        self.width = width
        self.height = height
        self.seed = seed
        self.safety_tolerance = safety_tolerance
        self.output_format = output_format

    def build_payload(self) -> dict:
        payload = {
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "safety_tolerance": self.safety_tolerance,
            "output_format": self.output_format,
        }
        if self.seed is not None and self.seed >= 0:
            payload["seed"] = self.seed
        if self.input_images:
            for index, image in enumerate(self.input_images, start=1):
                key = "input_image" if index == 1 else f"input_image_{index}"
                payload[key] = image
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return ["prompt", "input_image", "input_image_2", "input_image_3", "input_image_4",
                "input_image_5", "input_image_6", "input_image_7", "input_image_8",
                "seed", "width", "height", "safety_tolerance", "output_format"]

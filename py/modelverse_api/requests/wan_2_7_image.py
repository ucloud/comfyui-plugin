from typing import Optional, List
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor


class Wan27Image(BaseRequest):
    """
    Wan 2.7 Image text-to-image and image editing.

    Endpoint: POST /v1/images/generations
    """
    API_PATH = "/v1/images/generations"
    model: str = Field(default="wan2.7-image",
                       description="wan2.7-image or wan2.7-image-pro.")
    prompt: Optional[str] = Field(
        default=None, description="The prompt to generate or edit an image.")
    images: Optional[list[Tensor]] = Field(
        default=None, description="Reference images for image editing.")
    size: Optional[str] = Field(
        default="2K", description="Output size: 1K, 2K, 4K or <width>*<height>.")
    num_images: Optional[int] = Field(
        default=1, description="Number of images to generate (1 to 12).")
    seed: Optional[int] = Field(
        default=-1, description="Random seed, 0..2147483647.")
    watermark: Optional[bool] = Field(
        default=False, description="Whether to add a watermark.")
    thinking_mode: Optional[bool] = Field(
        default=None, description="Whether to enable thinking mode.")
    enable_sequential: Optional[bool] = Field(
        default=None, description="Whether to enable sequential image generation.")

    def __init__(
            self,
            prompt: Optional[str] = None,
            model: Optional[str] = "wan2.7-image",
            images: Optional[List[Tensor]] = None,
            size: Optional[str] = "2K",
            num_images: Optional[int] = 1,
            seed: Optional[int] = -1,
            watermark: Optional[bool] = False,
            thinking_mode: Optional[bool] = None,
            enable_sequential: Optional[bool] = None,
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt
        self.images = [image_to_base64(image)
                       for image in images] if images else None
        self.size = size
        self.num_images = num_images
        self.seed = seed
        self.watermark = watermark
        self.thinking_mode = thinking_mode
        self.enable_sequential = enable_sequential

    def build_payload(self) -> dict:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "image": self.images[0] if self.images else None,
            "images": self.images if self.images and len(self.images) > 1 else None,
            "size": self.size,
            "n": self.num_images,
            "seed": self.seed if self.seed is not None and self.seed >= 0 else None,
            "watermark": self.watermark,
            "thinking_mode": self.thinking_mode,
            "enable_sequential": self.enable_sequential,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["model"]

    def field_order(self):
        return ["model", "prompt", "images", "size", "n", "seed", "watermark",
                "thinking_mode", "enable_sequential"]

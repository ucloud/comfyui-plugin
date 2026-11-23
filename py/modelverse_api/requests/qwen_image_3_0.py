from typing import Optional, List
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor


class QwenImage30(BaseRequest):
    """
    Qwen Image 3.0 / 3.0 Pro text-to-image and image editing.

    Endpoint: POST /v1/images/generations
    """
    API_PATH = "/v1/images/generations"
    model: str = Field(default="qwen-image-3.0",
                       description="qwen-image-3.0 or qwen-image-3.0-pro.")
    prompt: Optional[str] = Field(
        default=None, description="The prompt to generate or edit an image.")
    images: Optional[list[Tensor]] = Field(
        default=None, description="Reference images, 1 to 3.")
    size: Optional[str] = Field(
        default="1024*1024", description="Output size, e.g. 1024*1024.")
    num_images: Optional[int] = Field(
        default=1, description="Number of images to generate (1 to 6).")
    seed: Optional[int] = Field(
        default=-1, description="Random seed, 0..2147483647.")
    negative_prompt: Optional[str] = Field(
        default=None, description="Negative prompt.")
    watermark: Optional[bool] = Field(
        default=False, description="Whether to add a watermark.")
    prompt_extend: Optional[bool] = Field(
        default=True, description="Whether to enable prompt extension.")

    def __init__(
            self,
            prompt: Optional[str] = None,
            model: Optional[str] = "qwen-image-3.0",
            images: Optional[List[Tensor]] = None,
            size: Optional[str] = "1024*1024",
            num_images: Optional[int] = 1,
            seed: Optional[int] = -1,
            negative_prompt: Optional[str] = None,
            watermark: Optional[bool] = False,
            prompt_extend: Optional[bool] = True,
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt
        self.images = [image_to_base64(image)
                       for image in images] if images else None
        self.size = size
        self.num_images = num_images
        self.seed = seed
        self.negative_prompt = negative_prompt
        self.watermark = watermark
        self.prompt_extend = prompt_extend

    def build_payload(self) -> dict:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "image": self.images[0] if self.images else None,
            "images": self.images if self.images and len(self.images) > 1 else None,
            "size": self.size,
            "n": self.num_images,
            "seed": self.seed if self.seed is not None and self.seed >= 0 else None,
            "negative_prompt": self.negative_prompt,
            "watermark": self.watermark,
            "prompt_extend": self.prompt_extend,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["model"]

    def field_order(self):
        return ["model", "prompt", "images", "size", "n", "seed",
                "negative_prompt", "watermark", "prompt_extend"]

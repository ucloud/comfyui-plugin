from typing import Optional
from pydantic import Field
from ..utils import BaseRequest


class GrokImagine(BaseRequest):
    """
    Grok Imagine image generation.

    Endpoint: POST /v1/images/generations
    """
    API_PATH = "/v1/images/generations"
    model: str = Field(default="grok-imagine-image",
                       description="grok-imagine-image or grok-imagine-image-quality.")
    prompt: str = Field(..., description="The prompt to generate an image from.")
    num_images: Optional[int] = Field(
        default=1, description="Number of images to generate (1 to 10).")
    size: Optional[str] = Field(
        default="1k", description="Output resolution: 1k or 2k.")
    aspect_ratio: Optional[str] = Field(
        default="4:3", description="Output aspect ratio.")

    def __init__(
            self,
            prompt: str,
            model: Optional[str] = "grok-imagine-image",
            num_images: Optional[int] = 1,
            size: Optional[str] = "1k",
            aspect_ratio: Optional[str] = "4:3",
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt
        self.num_images = num_images
        self.size = size
        self.aspect_ratio = aspect_ratio

    def build_payload(self) -> dict:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "n": self.num_images,
            "size": self.size,
            "aspect_ratio": self.aspect_ratio,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["model", "prompt"]

    def field_order(self):
        return ["model", "prompt", "n", "size", "aspect_ratio"]

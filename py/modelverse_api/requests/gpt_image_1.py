from typing import Optional
from pydantic import Field
from ..utils import BaseRequest


class GPTImage1(BaseRequest):
    """
    gpt-image-1 text-to-image via /v1/images/generations.
    """

    API_PATH = "/v1/images/generations"

    prompt: str = Field(..., description="Prompt text")
    num_images: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images (n)")
    size: Optional[str] = Field(default="1024x1024", description="Output size: 1024x1024, 1024x1536, 1536x1024")
    quality: Optional[str] = Field(default=None, description="Image quality: low, medium, high")
    output_format: Optional[str] = Field(default="png", description="Output format: png or jpeg")
    output_compression: Optional[int] = Field(default=100, ge=0, le=100, description="Compression strength 0-100")
    # negative_prompt and response_format removed per API behavior

    def __init__(
        self,
        prompt: str,
        num_images: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        quality: Optional[str] = None,
        output_format: Optional[str] = "png",
        output_compression: Optional[int] = 100,
        # guidance_scale removed per API behavior
        # negative_prompt/response_format removed per API behavior
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.num_images = num_images
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.output_compression = output_compression

    def build_payload(self) -> dict:
        payload = {
            "model": "gpt-image-1",
            "prompt": self.prompt,
            "n": self.num_images,
            "size": self.size,
            "quality": self.quality,
            "output_format": self.output_format,
            "output_compression": self.output_compression,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return [
            "model",
            "prompt",
            "n",
            "size",
            "quality",
            "output_format",
            "output_compression",
        ]

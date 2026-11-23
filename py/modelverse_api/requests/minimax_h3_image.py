from typing import Optional, List
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor


class MiniMaxH3ImageGeneration(BaseRequest):
    """
    MiniMax H3 Image text-to-image generation.

    Endpoint: POST /v1/images/generations
    """
    API_PATH = "/v1/images/generations"
    model: str = Field(default="minimax-h3-image", description="Model name.")
    prompt: str = Field(..., description="The prompt to generate an image from.")
    num_images: Optional[int] = Field(
        default=1, description="Number of images to generate (1 to 10).")
    size: Optional[str] = Field(
        default="auto", description="Output size, e.g. auto or <width>x<height>.")
    aspect_ratio: Optional[str] = Field(
        default=None, description="Output aspect ratio.")
    area: Optional[str] = Field(
        default=None, description="Output area: 1K, 2K or 4K.")
    output_format: Optional[str] = Field(
        default="jpeg", description="Output format: png, jpeg, jpg or webp.")
    response_format: Optional[str] = Field(
        default="url", description="Response format: url or b64_json.")
    aigc_watermark: Optional[bool] = Field(
        default=False, description="Whether to add an AIGC watermark.")

    def __init__(
            self,
            prompt: str,
            model: Optional[str] = "minimax-h3-image",
            num_images: Optional[int] = 1,
            size: Optional[str] = "auto",
            aspect_ratio: Optional[str] = None,
            area: Optional[str] = None,
            output_format: Optional[str] = "jpeg",
            response_format: Optional[str] = "url",
            aigc_watermark: Optional[bool] = False,
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt
        self.num_images = num_images
        self.size = size
        self.aspect_ratio = aspect_ratio
        self.area = area
        self.output_format = output_format
        self.response_format = response_format
        self.aigc_watermark = aigc_watermark

    def build_payload(self) -> dict:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "n": self.num_images,
            "size": self.size,
            "aspect_ratio": self.aspect_ratio,
            "area": self.area,
            "output_format": self.output_format,
            "response_format": self.response_format,
            "aigc_watermark": self.aigc_watermark,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["model", "prompt"]

    def field_order(self):
        return ["model", "prompt", "n", "size", "aspect_ratio", "area",
                "output_format", "response_format", "aigc_watermark"]


class MiniMaxH3ImageEdit(BaseRequest):
    """
    MiniMax H3 Image editing.

    Endpoint: POST /v1/images/edits
    """
    API_PATH = "/v1/images/edits"
    model: str = Field(default="minimax-h3-image", description="Model name.")
    prompt: str = Field(..., description="The edit prompt.")
    images: Optional[list[Tensor]] = Field(
        default=None, description="Reference images for editing.")
    num_images: Optional[int] = Field(
        default=1, description="Number of images to generate (1 to 10).")
    size: Optional[str] = Field(
        default="auto", description="Output size, e.g. auto or <width>x<height>.")
    aspect_ratio: Optional[str] = Field(
        default=None, description="Output aspect ratio.")
    area: Optional[str] = Field(
        default=None, description="Output area: 1K, 2K or 4K.")
    output_format: Optional[str] = Field(
        default="jpeg", description="Output format: png, jpeg, jpg or webp.")
    response_format: Optional[str] = Field(
        default="url", description="Response format: url or b64_json.")
    aigc_watermark: Optional[bool] = Field(
        default=False, description="Whether to add an AIGC watermark.")

    def __init__(
            self,
            prompt: str,
            images: Optional[List[Tensor]] = None,
            model: Optional[str] = "minimax-h3-image",
            num_images: Optional[int] = 1,
            size: Optional[str] = "auto",
            aspect_ratio: Optional[str] = None,
            area: Optional[str] = None,
            output_format: Optional[str] = "jpeg",
            response_format: Optional[str] = "url",
            aigc_watermark: Optional[bool] = False,
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt
        self.images = [image_to_base64(image)
                       for image in images] if images else None
        self.num_images = num_images
        self.size = size
        self.aspect_ratio = aspect_ratio
        self.area = area
        self.output_format = output_format
        self.response_format = response_format
        self.aigc_watermark = aigc_watermark

    def build_payload(self) -> dict:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "image": self.images[0] if self.images else None,
            "images": self.images[1:] if self.images and len(self.images) > 1 else None,
            "n": self.num_images,
            "size": self.size,
            "aspect_ratio": self.aspect_ratio,
            "area": self.area,
            "output_format": self.output_format,
            "response_format": self.response_format,
            "aigc_watermark": self.aigc_watermark,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["model", "prompt", "image"]

    def field_order(self):
        return ["model", "prompt", "image", "n", "size", "aspect_ratio", "area",
                "output_format", "response_format", "aigc_watermark"]

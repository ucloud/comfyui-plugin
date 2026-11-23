from typing import Optional, List
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor


class DoubaoSeedream(BaseRequest):
    """
    Doubao Seedream image generation / reference-image editing.

    Endpoint: POST /v1/images/generations
    """
    API_PATH = "/v1/images/generations"
    model: str = Field(default="doubao-seedream-5-5-261028",
                       description="Doubao Seedream model name.")
    prompt: str = Field(..., description="The prompt to generate an image from.")
    images: Optional[list[Tensor]] = Field(
        default=None, description="Reference images for image-to-image.")
    size: Optional[str] = Field(
        default="2K", description="Output size, e.g. 1K/2K/4K/auto or <width>x<height>.")
    output_format: Optional[str] = Field(
        default=None, description="Output format: png or jpeg.")
    watermark: Optional[bool] = Field(
        default=None, description="Whether to add an AI generated watermark.")
    response_format: Optional[str] = Field(
        default="url", description="Response format: url or b64_json.")

    def __init__(
            self,
            prompt: str,
            model: Optional[str] = "doubao-seedream-5-5-261028",
            images: Optional[List[Tensor]] = None,
            size: Optional[str] = "2K",
            output_format: Optional[str] = None,
            watermark: Optional[bool] = None,
            response_format: Optional[str] = "url",
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.prompt = prompt
        self.images = [image_to_base64(image)
                       for image in images] if images else None
        self.size = size
        self.output_format = output_format
        self.watermark = watermark
        self.response_format = response_format

    def build_payload(self) -> dict:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": self.images,
            "size": self.size,
            "output_format": self.output_format,
            "watermark": self.watermark,
            "response_format": self.response_format,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["model", "prompt"]

    def field_order(self):
        return ["model", "prompt", "images", "size", "output_format", "watermark", "response_format"]

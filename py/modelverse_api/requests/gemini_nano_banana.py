from typing import Optional, List, Dict, Any
from pydantic import Field
from ..utils import BaseRequest
from torch import Tensor
import base64
import io
from PIL import Image
import numpy as np


def _tensor_to_base64(image: Tensor, mime_type: str = "image/png") -> Dict[str, str]:
    """Convert a ComfyUI image tensor (HWC or 4D with batch) to base64 without data URI."""
    if image is None:
        return None

    if hasattr(image, 'shape') and len(image.shape) == 4:
        image = image[0]

    np_img = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)

    fmt = 'PNG' if mime_type.lower().endswith('png') else 'JPEG'
    with io.BytesIO() as bio:
        pil_img.save(bio, format=fmt)
        data = bio.getvalue()

    return {
        "mimeType": mime_type,
        "data": base64.b64encode(data).decode("utf-8"),
    }


class GeminiNanoBanana21Request(BaseRequest):
    """
    Request builder for Gemini Nano Banana 2.1 generateContent endpoint.
    Supports text-to-image and image-edit with 1K/2K/4K output resolution.
    """

    API_PATH_TEMPLATE = "/v1beta/models/{model}:generateContent"

    prompt: str = Field(..., description="Text prompt")
    model: str = Field(default="gemini-nano-banana-2.1", description="Gemini Nano Banana model")
    image: Optional[Tensor] = Field(default=None, description="Optional input image for edit")
    image_size: str = Field(default="1K", description="Output resolution: 1K, 2K or 4K")
    aspect_ratio: str = Field(default="1:1", description="Output aspect ratio")
    mime_type: str = Field(default="image/png", description="MIME type for inline image data")

    def __init__(
            self,
            prompt: str,
            model: str = "gemini-nano-banana-2.1",
            image: Optional[Tensor] = None,
            image_size: str = "1K",
            aspect_ratio: str = "1:1",
            mime_type: str = "image/png",
            **kwargs):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.model = model
        self.API_PATH = self.API_PATH_TEMPLATE.format(model=model)
        self.image = image
        self.image_size = image_size
        self.aspect_ratio = aspect_ratio
        self.mime_type = mime_type

    def build_payload(self) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []
        if self.prompt:
            parts.append({"text": self.prompt})

        if isinstance(self.image, Tensor):
            parts.append({
                "inlineData": _tensor_to_base64(self.image, self.mime_type)
            })

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": self.aspect_ratio,
                    "imageSize": self.image_size,
                },
            },
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return ["model", "prompt", "image", "image_size", "aspect_ratio", "mime_type"]

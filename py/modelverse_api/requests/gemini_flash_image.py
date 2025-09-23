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

    # If batch, take first frame
    if hasattr(image, 'shape') and len(image.shape) == 4:
        image = image[0]

    # Tensor (H, W, C) in 0..1 -> uint8
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


class GeminiFlashImageRequest(BaseRequest):
    """
    Request builder for Gemini 2.5 Flash Image generateContent endpoint.
    Supports text-to-image and image-edit (text+image to image).
    """

    API_PATH = "/v1beta/models/gemini-2.5-flash-image:generateContent"

    prompt: str = Field(..., description="Text prompt")
    image: Optional[Tensor] = Field(default=None, description="Optional input image for edit")
    mime_type: str = Field(default="image/png", description="MIME type for inline image data")

    def __init__(self, prompt: str, image: Optional[Tensor] = None, mime_type: str = "image/png", **kwargs):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.image = image
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
            },
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return ["prompt", "image", "mime_type"]


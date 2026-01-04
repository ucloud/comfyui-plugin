"""
Gemini 3 Pro Image Preview (Nano Banana Pro) request builder.
Supports text-to-image and image editing with advanced features like
aspect ratio, resolution control, and Google Search grounding.
"""
import io
import base64
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from pydantic import Field
from torch import Tensor

from ..utils import BaseRequest


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


class GeminiProImageRequest(BaseRequest):
    """
    Request builder for Gemini 3 Pro Image Preview (Nano Banana Pro) generateContent endpoint.
    Supports text-to-image, image editing, and advanced features like aspect ratio,
    resolution control (1K/2K/4K), and Google Search grounding.
    """

    API_PATH = "/v1beta/models/gemini-3-pro-image-preview:generateContent"

    prompt: str = Field(..., description="Text prompt")
    images: Optional[List[Tensor]] = Field(default=None, description="Optional input images (up to 14)")
    mime_type: str = Field(default="image/png", description="MIME type for inline image data")
    aspect_ratio: Optional[str] = Field(default=None, description="Aspect ratio: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9")
    image_size: Optional[str] = Field(default=None, description="Image resolution: 1K, 2K, 4K")
    use_google_search: bool = Field(default=False, description="Enable Google Search grounding")

    def __init__(
        self,
        prompt: str,
        images: Optional[List[Tensor]] = None,
        mime_type: str = "image/png",
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        use_google_search: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.images = images
        self.mime_type = mime_type
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size
        self.use_google_search = use_google_search

    def build_payload(self) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []
        
        # Add text prompt
        if self.prompt:
            parts.append({"text": self.prompt})

        # Add images (up to 14)
        if self.images:
            for img in self.images[:14]:  # Limit to 14 images
                if isinstance(img, Tensor):
                    parts.append({
                        "inlineData": _tensor_to_base64(img, self.mime_type)
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

        # Add imageConfig if aspect_ratio or image_size specified
        if self.aspect_ratio or self.image_size:
            image_config = {}
            if self.aspect_ratio:
                image_config["aspectRatio"] = self.aspect_ratio
            if self.image_size:
                image_config["imageSize"] = self.image_size
            payload["generationConfig"]["imageConfig"] = image_config

        # Add Google Search tool if enabled
        if self.use_google_search:
            payload["tools"] = [{"google_search": {}}]

        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return ["prompt", "images", "mime_type", "aspect_ratio", "image_size", "use_google_search"]

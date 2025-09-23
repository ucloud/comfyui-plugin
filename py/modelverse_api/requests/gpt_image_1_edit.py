from typing import Optional, Tuple, Dict, Any
from pydantic import Field
from ..utils import BaseRequest, tensor2images
from torch import Tensor
import io
from PIL import Image


def _tensor_to_png_file(image: Tensor, filename: str) -> Tuple[str, bytes, str] | None:
    """Convert a ComfyUI tensor image (3D or 4D) to a PNG file tuple for requests files."""
    if image is None:
        return None
    # Keep batch if present; tensor2images expects a batch dimension
    try:
        if hasattr(image, 'shape') and len(image.shape) == 3:
            # add batch dim
            img_batch = image.unsqueeze(0)
        else:
            img_batch = image
        pil_img = tensor2images(img_batch)[0]
    except Exception:
        # Fallback: attempt naive conversion
        from PIL import Image as _Image
        import numpy as _np
        arr = (image.cpu().numpy() * 255.0).clip(0, 255).astype('uint8')
        pil_img = _Image.fromarray(arr)
    with io.BytesIO() as bio:
        pil_img.save(bio, format="PNG")
        data = bio.getvalue()
    return (filename, data, "image/png")


class GPTImage1Edit(BaseRequest):
    """
    gpt-image-1 image edit via /v1/images/edits using multipart/form-data.
    """

    API_PATH = "/v1/images/edits"

    prompt: str = Field(..., description="Edit prompt")
    image: Tensor = Field(..., description="Input image")
    mask: Optional[Tensor] = Field(default=None, description="Optional mask image")

    num_images: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images (n)")
    size: Optional[str] = Field(default="1024x1024", description="Output size: 1024x1024, 1024x1536, 1536x1024")
    quality: Optional[str] = Field(default=None, description="Image quality: low, medium, high")
    output_format: Optional[str] = Field(default="png", description="Output format: png or jpeg")
    output_compression: Optional[int] = Field(default=100, ge=0, le=100, description="Compression strength 0-100")

    def __init__(
        self,
        prompt: str,
        image: Tensor,
        mask: Optional[Tensor] = None,
        num_images: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        quality: Optional[str] = None,
        output_format: Optional[str] = "png",
        output_compression: Optional[int] = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.image = image
        self.mask = mask
        self.num_images = num_images
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.output_compression = output_compression

    def build_multipart(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = {
            "model": "gpt-image-1",
            "prompt": self.prompt,
            "n": self.num_images,
            "size": self.size,
            "output_format": self.output_format,
            "output_compression": str(self.output_compression) if self.output_compression is not None else None,
        }
        if self.quality:
            data["quality"] = self.quality
        # Filter None values
        data = {k: v for k, v in data.items() if v is not None and v != ""}

        files = {}
        img_file = _tensor_to_png_file(self.image, "image.png")
        if img_file is not None:
            files["image"] = img_file
        if self.mask is not None:
            mask_file = _tensor_to_png_file(self.mask, "mask.png")
            if mask_file is not None:
                files["mask"] = mask_file

        return data, files

    # For interface completeness; not used by multipart path
    def build_payload(self) -> dict:
        return {}

    def field_required(self):
        return ["prompt", "image"]

    def field_order(self):
        return [
            "model",
            "prompt",
            "image",
            "mask",
            "n",
            "size",
            "quality",
            "output_format",
            "output_compression",
        ]

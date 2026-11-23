from typing import Optional, Tuple, Dict, Any, List
from pydantic import Field
from ..utils import BaseRequest, tensor2images
from torch import Tensor
import io
from PIL import Image


def _tensor_to_png_file(image: Tensor, filename: str) -> Optional[Tuple[str, bytes, str]]:
    """Convert a ComfyUI tensor image (3D or 4D) to a PNG file tuple for requests files."""
    if image is None:
        return None
    try:
        if hasattr(image, 'shape') and len(image.shape) == 3:
            img_batch = image.unsqueeze(0)
        else:
            img_batch = image
        pil_img = tensor2images(img_batch)[0]
    except Exception:
        from PIL import Image as _Image
        import numpy as _np
        arr = (image.cpu().numpy() * 255.0).clip(0, 255).astype('uint8')
        pil_img = _Image.fromarray(arr)
    with io.BytesIO() as bio:
        pil_img.save(bio, format="PNG")
        data = bio.getvalue()
    return (filename, data, "image/png")


class GPTImage2Generate(BaseRequest):
    """
    gpt-image-2 family text-to-image via /v1/images/generations.
    """

    API_PATH = "/v1/images/generations"

    model: str = Field(default="gpt-image-2", description="gpt-image model name")
    prompt: str = Field(..., description="Text prompt")
    num_images: Optional[int] = Field(default=1, description="Number of images (n), 1..10")
    size: Optional[str] = Field(default="1024x1024", description="Output size, e.g. auto or <width>x<height>")
    quality: Optional[str] = Field(default=None, description="Image quality: low, medium, high, xhigh, max")
    output_format: Optional[str] = Field(default="png", description="Output format: png or jpeg")
    output_compression: Optional[int] = Field(default=100, description="Compression strength 0-100 (jpeg only)")

    def __init__(
        self,
        prompt: str,
        model: Optional[str] = "gpt-image-2",
        num_images: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        quality: Optional[str] = None,
        output_format: Optional[str] = "png",
        output_compression: Optional[int] = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.prompt = prompt
        self.num_images = num_images
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.output_compression = output_compression

    def build_payload(self) -> dict:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "n": self.num_images,
            "size": self.size,
            "quality": self.quality,
            "output_format": self.output_format,
            "output_compression": self.output_compression if self.output_format == "jpeg" else None,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return ["model", "prompt", "n", "size", "quality", "output_format", "output_compression"]


class GPTImage2Edit(BaseRequest):
    """
    gpt-image-2 family image edit via /v1/images/edits using multipart/form-data.
    """

    API_PATH = "/v1/images/edits"

    model: str = Field(default="gpt-image-2", description="gpt-image model name")
    prompt: str = Field(..., description="Edit prompt")
    images: List[Tensor] = Field(default=None, description="Input images")
    mask: Optional[Tensor] = Field(default=None, description="Optional mask image")

    num_images: Optional[int] = Field(default=1, description="Number of images (n), 1..10")
    size: Optional[str] = Field(default="1024x1024", description="Output size, e.g. auto or <width>x<height>")
    quality: Optional[str] = Field(default=None, description="Image quality: low, medium, high, xhigh, max")
    output_format: Optional[str] = Field(default="png", description="Output format: png or jpeg")
    output_compression: Optional[int] = Field(default=100, description="Compression strength 0-100 (jpeg only)")

    def __init__(
        self,
        prompt: str,
        images: List[Tensor],
        model: Optional[str] = "gpt-image-2",
        mask: Optional[Tensor] = None,
        num_images: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        quality: Optional[str] = None,
        output_format: Optional[str] = "png",
        output_compression: Optional[int] = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.prompt = prompt
        self.images = images
        self.mask = mask
        self.num_images = num_images
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.output_compression = output_compression

    def build_multipart(self) -> Tuple[Dict[str, Any], Any]:
        data = {
            "model": self.model,
            "prompt": self.prompt,
            "n": self.num_images,
            "size": self.size,
            "output_format": self.output_format,
            "output_compression": str(self.output_compression) if self.output_format == "jpeg" else None,
        }
        if self.quality:
            data["quality"] = self.quality
        data = {k: v for k, v in data.items() if v is not None and v != ""}

        image_files = [_tensor_to_png_file(image, "image.png") for image in self.images or []]
        image_files = [f for f in image_files if f is not None]
        if not image_files:
            raise ValueError("At least one input image is required")

        if len(image_files) == 1:
            files = {"image": image_files[0]}
        else:
            files = [("image[]", f) for f in image_files]

        if self.mask is not None:
            mask_file = _tensor_to_png_file(self.mask, "mask.png")
            if mask_file is not None:
                if isinstance(files, list):
                    files.append(("mask", mask_file))
                else:
                    files["mask"] = mask_file

        return data, files

    # For interface completeness; not used by multipart path
    def build_payload(self) -> dict:
        return {}

    def field_required(self):
        return ["prompt", "images"]

    def field_order(self):
        return ["model", "prompt", "images", "mask", "n", "size",
                "quality", "output_format", "output_compression"]

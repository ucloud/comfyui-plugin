"""
Gemini 3 Pro Image Preview (Nano Banana Pro) ComfyUI Node.
Professional asset production with advanced features:
- High-resolution output (1K, 2K, 4K)
- Aspect ratio control
- Google Search grounding
- Up to 14 reference images
"""
import base64
import torch
from typing import Optional, List, Dict, Any
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.gemini_pro_image import GeminiProImageRequest
from .modelverse_api.utils import decode_image, images2tensor


def _extract_images_from_gemini_response(resp: Dict[str, Any]) -> List[torch.Tensor]:
    """Extract image tensors from Gemini API response."""
    tensors: List[torch.Tensor] = []
    candidates = resp.get("candidates", [])
    for cand in candidates:
        content = cand.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            inline = part.get("inlineData")
            if inline:
                data_b64 = inline.get("data")
                if data_b64:
                    raw = base64.b64decode(data_b64)
                    pil_img = decode_image(raw)
                    tensors.append(images2tensor(pil_img))
            # Also print any text response
            if part.get("text"):
                print("INFO: Gemini Pro Image text response:", part.get("text"))
    return tensors


# Aspect ratio options
ASPECT_RATIOS = ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
# Resolution options
IMAGE_SIZES = ["1K", "2K", "4K"]


class GeminiProImageNode:
    """
    Gemini 3 Pro Image Preview (Nano Banana Pro) via Modelverse API.
    
    Professional asset production with:
    - High-resolution output (1K, 2K, 4K)
    - Aspect ratio control
    - Google Search grounding for real-time info
    - Up to 14 reference images
    
    Endpoint: /v1beta/models/gemini-3-pro-image-preview:generateContent
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "Create a professional product photo"}),
                "mime_type": (["image/png", "image/jpeg"], {"default": "image/png"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto"}),
                "image_size": (IMAGE_SIZES, {"default": "1K"}),
                "use_google_search": (IO.BOOLEAN, {"default": False, "tooltip": "Enable Google Search grounding for real-time info"}),
                "num_requests": (IO.INT, {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number"}),
            },
            "optional": {
                "image1": (IO.IMAGE, {"default": None, "tooltip": "Reference image 1"}),
                "image2": (IO.IMAGE, {"default": None, "tooltip": "Reference image 2"}),
                "image3": (IO.IMAGE, {"default": None, "tooltip": "Reference image 3"}),
                "image4": (IO.IMAGE, {"default": None, "tooltip": "Reference image 4"}),
                "image5": (IO.IMAGE, {"default": None, "tooltip": "Reference image 5"}),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "execute"

    async def execute(
        self,
        client,
        prompt: str,
        mime_type: str = "image/png",
        aspect_ratio: str = "auto",
        image_size: str = "1K",
        use_google_search: bool = False,
        num_requests: int = 1,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
    ):
        if not prompt:
            raise ValueError("Prompt is required")

        mv_client = ModelverseClient(client["api_key"])

        # Collect input images
        images = []
        for img in [image1, image2, image3, image4, image5]:
            if img is not None:
                images.append(img)

        # Process aspect_ratio
        ar = aspect_ratio if aspect_ratio != "auto" else None

        outputs: List[torch.Tensor] = []
        for i in range(num_requests):
            req = GeminiProImageRequest(
                prompt=prompt,
                images=images if images else None,
                mime_type=mime_type,
                aspect_ratio=ar,
                image_size=image_size,
                use_google_search=use_google_search,
            )
            payload = req.build_payload()
            resp = mv_client.post(req.API_PATH, payload)

            if isinstance(resp, dict) and resp.get("error"):
                err = resp.get("error")
                raise Exception(f"GeminiProImage error: {err.get('message', 'Unknown error')}")

            tensors = _extract_images_from_gemini_response(resp)
            if not tensors:
                print("WARN:", "No image data found in Gemini Pro response; check console for details.")
                continue
            outputs.append(tensors[0])

        if not outputs:
            raise Exception("No images generated from Gemini 3 Pro Image Preview")

        return (torch.cat(outputs, dim=0),)


NODE_CLASS_MAPPINGS = {
    'Gemini Pro Image (Nano Banana Pro)': GeminiProImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Gemini Pro Image (Nano Banana Pro)': 'Gemini 3 Pro Image Preview',
}

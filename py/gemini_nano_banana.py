import base64
from typing import List, Dict, Any
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.gemini_nano_banana import GeminiNanoBanana21Request
from .modelverse_api.utils import decode_image, images2tensor


MODEL = "gemini-nano-banana-2.1"

ASPECT_RATIOS = [
    "1:1", "3:2", "2:3", "3:4", "1:4", "4:1", "4:3", "4:5",
    "5:4", "1:8", "8:1", "9:16", "16:9", "21:9", "9:21",
]

IMAGE_SIZES = ["1K", "2K", "4K"]


def _extract_images_from_gemini_response(resp: Dict[str, Any]) -> List[torch.Tensor]:
    """Parse Gemini API response and return list of tensors for any inline images."""
    images = []
    try:
        candidates = resp.get("candidates", []) or []
        for cand in candidates:
            content = cand.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                inline = part.get("inlineData")
                if inline and isinstance(inline, dict):
                    data_b64 = inline.get("data")
                    if data_b64:
                        try:
                            img_bytes = base64.b64decode(data_b64)
                            pil_img = decode_image(img_bytes)
                            images.append(pil_img)
                        except Exception:
                            continue
    except Exception:
        pass
    if not images:
        return []
    return [images2tensor(images)]


class GeminiNanoBanana21Node:
    """
    Gemini Nano Banana 2.1 text-to-image and image editing.

    Endpoint: /v1beta/models/gemini-nano-banana-2.1:generateContent
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the image or the edit",
                }),
                "image_size": (IMAGE_SIZES, {
                    "default": "1K",
                    "tooltip": "Output resolution: 1K, 2K or 4K",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "1:1",
                    "tooltip": "Output aspect ratio",
                }),
                "mime_type": (["image/png", "image/jpeg"], {
                    "default": "image/png",
                    "tooltip": "MIME type of the input/output inline image data",
                }),
                "num_requests": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of requests to make (1 to 10)",
                }),
            },
            "optional": {
                "image": (IO.IMAGE, {
                    "default": None,
                    "tooltip": "Optional input image for editing",
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE/Gemini"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      prompt: str,
                      image_size: str = "1K",
                      aspect_ratio: str = "1:1",
                      mime_type: str = "image/png",
                      num_requests: int = 1,
                      image=None):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")
        if not prompt:
            raise ValueError("Prompt is required")

        mv_client = ModelverseClient(api_key)
        # Gemini native endpoints authenticate via x-goog-api-key.
        mv_client.headers["x-goog-api-key"] = api_key

        outputs: List[torch.Tensor] = []
        for _ in range(num_requests):
            req = GeminiNanoBanana21Request(
                prompt=prompt,
                model=MODEL,
                image=image,
                image_size=image_size,
                aspect_ratio=aspect_ratio,
                mime_type=mime_type,
            )
            payload = req.build_payload()
            resp = mv_client.post(req.API_PATH, payload)

            if isinstance(resp, dict) and resp.get("error"):
                err = resp.get("error")
                raise Exception(f"GeminiNanoBanana error: {err.get('message', 'Unknown error')}")

            tensors = _extract_images_from_gemini_response(resp)
            if not tensors:
                print("WARN:", "No image data found in Gemini response; check console for details.")
                continue
            outputs.append(tensors[0])

        if not outputs:
            return (torch.zeros((1, 3, 1, 1)),)

        return (torch.cat(outputs, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse GeminiNanoBanana21Node": GeminiNanoBanana21Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse GeminiNanoBanana21Node": "Modelverse Gemini Nano Banana 2.1",
}

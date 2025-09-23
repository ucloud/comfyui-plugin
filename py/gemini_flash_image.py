import base64
import torch
from typing import Optional, List, Dict, Any
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.gemini_flash_image import GeminiFlashImageRequest
from .modelverse_api.utils import decode_image, images2tensor


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


class GeminiFlashImageNode:
    """
    Gemini 2.5 Flash Image (text-to-image and image-edit) via Modelverse API.

    Endpoint: /v1beta/models/gemini-2.5-flash-image:generateContent
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme"}),
                "mime_type": (["image/png", "image/jpeg"], {"default": "image/png"}),
                "num_requests": (IO.INT, {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number"}),
            },
            "optional": {
                "image": (IO.IMAGE, {"default": None, "tooltip": "Optional input image for edit"}),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      prompt: str,
                      mime_type: str = "image/png",
                      num_requests: int = 1,
                      image=None):
        if not prompt:
            raise ValueError("Prompt is required")

        mv_client = ModelverseClient(client["api_key"])

        outputs: List[torch.Tensor] = []
        for i in range(num_requests):
            req = GeminiFlashImageRequest(prompt=prompt, image=image, mime_type=mime_type)
            payload = req.build_payload()
            resp = mv_client.post(req.API_PATH, payload)

            if isinstance(resp, dict) and resp.get("error"):
                err = resp.get("error")
                raise Exception(f"GeminiFlashImage error: {err.get('message', 'Unknown error')}")

            tensors = _extract_images_from_gemini_response(resp)
            if not tensors:
                print("WARN:", "No image data found in Gemini response; check console for details.")
                continue
            # tensors is a list with one batch tensor; append that tensor
            outputs.append(tensors[0])

        if not outputs:
            # Return an empty black image tensor (1x3x1x1) to avoid breaking graph
            return (torch.zeros((1, 3, 1, 1)),)

        # Concatenate along batch dimension
        return (torch.cat(outputs, dim=0),)


NODE_CLASS_MAPPINGS = {
    "NanoBanana": GeminiFlashImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana": "Modelverse Gemini 2.5 Flash Image",
}

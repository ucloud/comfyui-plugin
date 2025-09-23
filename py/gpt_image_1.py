import base64
import torch
from typing import List
from comfy.comfy_types.node_typing import IO

from .modelverse_api.utils import imageurl2tensor, decode_image, images2tensor
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.gpt_image_1 import GPTImage1


class GPTImage1Node:
    """
    gpt-image-1 text-to-image via Modelverse /v1/images/generations API.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "a beautiful flower"}),
                "size": (["1024x1024", "1024x1536", "1536x1024"], {"default": "1024x1024"}),
                "num_requests": (IO.INT, {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number"}),
                "num_images": (IO.INT, {"default": 1, "min": 1, "max": 4, "step": 1, "display": "number"}),
            },
            "optional": {
                "quality": (["", "low", "medium", "high"], {"default": ""}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "output_compression": (IO.INT, {"default": 100, "min": 0, "max": 100, "step": 1, "display": "number"}),
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
        size: str = "1024x1024",
        num_requests: int = 1,
        num_images: int = 1,
        quality: str = "",
        output_format: str = "png",
        output_compression: int = 100,
    ):

        if not prompt:
            raise ValueError("Prompt is required")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [
            mv_client.async_send_request(
                GPTImage1(
                    prompt=prompt,
                    num_images=num_images,
                    size=size,
                    quality=quality if quality != "" else None,
                    output_format=output_format,
                    output_compression=output_compression,
                )
            )
            for i in range(num_requests)
        ]

        results = await mv_client.run_tasks(tasks)

        output_images_list: List[torch.Tensor] = []
        for data_list in results:
            if not data_list:
                print("WARN:", "No output in current request. Skipping...")
                continue

            # Auto-detect output type: prefer URL if present, else decode b64_json
            has_url = False
            try:
                for item in data_list:
                    if isinstance(item, dict) and item.get("url"):
                        has_url = True
                        break
            except Exception:
                has_url = False

            if has_url:
                try:
                    output_images = imageurl2tensor(data_list)
                except Exception:
                    print("WARN:", "Failed to load URL images; attempting b64_json decode.")
                    has_url = False  # fall through to b64 decode

            if not has_url:
                images = []
                for item in data_list:
                    if not isinstance(item, dict):
                        continue
                    b64v = item.get("b64_json") or item.get("b64")
                    if not b64v:
                        continue
                    if isinstance(b64v, str) and b64v.startswith("data:"):
                        try:
                            b64v = b64v.split(",", 1)[1]
                        except Exception:
                            pass
                    try:
                        img_bytes = base64.b64decode(b64v)
                        pil_img = decode_image(img_bytes)
                        images.append(pil_img)
                    except Exception:
                        continue

                if not images:
                    print("WARN:", "No decodable base64 image found.")
                    continue
                output_images = images2tensor(images)

            output_images_list.append(output_images)

        if not output_images_list:
            return (torch.zeros((1, 3, 1, 1)),)

        return (torch.cat(output_images_list, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse GPTImage1Node": GPTImage1Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse GPTImage1Node": "Modelverse GPT Image 1",
}

import base64
import torch
from typing import List
from comfy.comfy_types.node_typing import IO

from .modelverse_api.utils import imageurl2tensor, decode_image, images2tensor
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.qwen_image import QwenImage


class QwenImageT2INode:
    """
    Qwen/Qwen-Image text-to-image via Modelverse /v1/images/generations API.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "a beautiful flower"}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"], {"default": "1:1"}),
                "num_requests": (IO.INT, {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number"}),
                "num_images": (IO.INT, {"default": 1, "min": 1, "max": 4, "step": 1, "display": "number"}),
                "steps": (IO.INT, {"default": 20, "min": 1, "max": 50, "step": 1, "display": "number"}),
                "seed": (IO.INT, {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "guidance_scale": (IO.FLOAT, {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "display": "number"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
            },
            "optional": {
                "negative_prompt": (IO.STRING, {"multiline": True, "default": ""}),
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
        aspect_ratio: str = "1:1",
        num_requests: int = 1,
        num_images: int = 1,
        steps: int = 20,
        seed: int = -1,
        guidance_scale: float = 2.5,
        response_format: str = "url",
        negative_prompt: str = "",
    ):

        if not prompt:
            raise ValueError("Prompt is required")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [
            mv_client.async_send_request(
                QwenImage(
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    num_images=num_images,
                    seed=seed + i,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    response_format=response_format,
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

            if response_format == "url":
                output_images = imageurl2tensor(data_list)
            else:
                images = []
                for item in data_list:
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
    "Modelverse QwenImageT2INode": QwenImageT2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse QwenImageT2INode": "Modelverse Qwen Image",
}


import base64
from typing import List
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.flux_pro_1_1 import FluxPro11
from .modelverse_api.utils import decode_image, images2tensor


class FluxPro11Node:
    """
    Flux Pro 1.1 image generation.

    Endpoint: POST /v1/flux-pro-1.1
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the image",
                }),
                "width": (IO.INT, {
                    "default": 1024,
                    "min": 256,
                    "max": 1440,
                    "step": 32,
                    "display": "number",
                    "tooltip": "Image width, 256..1440 and a multiple of 32",
                }),
                "height": (IO.INT, {
                    "default": 1024,
                    "min": 256,
                    "max": 1440,
                    "step": 32,
                    "display": "number",
                    "tooltip": "Image height, 256..1440 and a multiple of 32",
                }),
                "num_requests": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of requests to make (1 to 10)",
                }),
                "seed": (IO.INT, {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible results. -1 for random seed",
                }),
            },
            "optional": {
                "image_prompt": (IO.IMAGE, {
                    "tooltip": "Optional image prompt (base64 encoded before sending)",
                    "forceInput": False,
                    "default": None,
                }),
                "prompt_upsampling": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "Automatically enhance the prompt for more creative results",
                }),
                "safety_tolerance": (IO.INT, {
                    "default": 2,
                    "min": 0,
                    "max": 6,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Moderation tolerance, 0 (strictest) to 6 (loosest)",
                }),
                "output_format": (["png", "jpg", "jpeg"], {
                    "default": "png",
                    "tooltip": "Output image format",
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE/Flux"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      prompt,
                      width=1024,
                      height=1024,
                      num_requests=1,
                      seed=-1,
                      image_prompt=None,
                      prompt_upsampling=False,
                      safety_tolerance=2,
                      output_format="png"):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")
        if width < 256 or width > 1440 or width % 32 != 0:
            raise ValueError("Width must be within 256..1440 and a multiple of 32")
        if height < 256 or height > 1440 or height % 32 != 0:
            raise ValueError("Height must be within 256..1440 and a multiple of 32")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [mv_client.async_send_request(FluxPro11(
            prompt=prompt,
            image_prompt=image_prompt,
            width=width,
            height=height,
            prompt_upsampling=prompt_upsampling,
            seed=seed + i,
            safety_tolerance=safety_tolerance,
            output_format=output_format,
        )) for i in range(num_requests)]

        results = await mv_client.run_tasks(tasks)

        output_images_list: List[torch.Tensor] = []
        for data_list in results:
            if not data_list:
                print("WARN:", "No output in current request. Skipping...")
                continue

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
                    images.append(decode_image(img_bytes))
                except Exception:
                    continue

            if not images:
                print("WARN:", "No decodable base64 image found.")
                continue
            output_images_list.append(images2tensor(images))

        if not output_images_list:
            return (torch.zeros((1, 3, 1, 1)),)

        print("INFO:", f"{len(output_images_list)}/{num_requests} request made successfully.")
        return (torch.cat(output_images_list, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse FluxPro11Node": FluxPro11Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse FluxPro11Node": "Modelverse Flux Pro 1.1",
}

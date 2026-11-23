import base64
from typing import List
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.flux_2_pro import Flux2Pro
from .modelverse_api.utils import decode_image, images2tensor


class Flux2ProNode:
    """
    Flux 2 Pro image generation / reference-image editing.

    Endpoint: POST /v1/flux-2-pro
    """

    @classmethod
    def INPUT_TYPES(s):
        required = {
            "client": ("MODELVERSE_API_CLIENT",),
            "prompt": (IO.STRING, {
                "multiline": True,
                "default": "",
                "tooltip": "Text prompt describing the image",
            }),
            "width": (IO.INT, {
                "default": 1024,
                "min": 64,
                "max": 4096,
                "step": 32,
                "display": "number",
                "tooltip": "Image width, >= 64 and a multiple of 32",
            }),
            "height": (IO.INT, {
                "default": 1024,
                "min": 64,
                "max": 4096,
                "step": 32,
                "display": "number",
                "tooltip": "Image height, >= 64 and a multiple of 32",
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
        }
        optional = {
            "safety_tolerance": (["0", "1", "2", "3", "4", "5", "6"], {
                "default": "2",
                "tooltip": "Moderation tolerance, 0 (strictest) to 6 (loosest)",
            }),
            "output_format": (["png", "jpg", "jpeg"], {
                "default": "png",
                "tooltip": "Output image format",
            }),
        }
        for i in range(1, 9):
            name = "input_image" if i == 1 else f"input_image_{i}"
            optional[name] = (IO.IMAGE, {
                "tooltip": f"Optional reference image {i}",
                "forceInput": False,
                "default": None,
            })
        return {"required": required, "optional": optional}

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
                      safety_tolerance="2",
                      output_format="png",
                      input_image=None,
                      input_image_2=None,
                      input_image_3=None,
                      input_image_4=None,
                      input_image_5=None,
                      input_image_6=None,
                      input_image_7=None,
                      input_image_8=None):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")
        if width < 64 or width % 32 != 0:
            raise ValueError("Width must be >= 64 and a multiple of 32")
        if height < 64 or height % 32 != 0:
            raise ValueError("Height must be >= 64 and a multiple of 32")

        input_images = [
            image for image in [
                input_image, input_image_2, input_image_3, input_image_4,
                input_image_5, input_image_6, input_image_7, input_image_8,
            ] if image is not None
        ]
        if len(input_images) > 8:
            raise ValueError("At most 8 input images are supported")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [mv_client.async_send_request(Flux2Pro(
            prompt=prompt,
            input_images=input_images if input_images else None,
            width=width,
            height=height,
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
    "Modelverse Flux2ProNode": Flux2ProNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse Flux2ProNode": "Modelverse Flux 2 Pro",
}

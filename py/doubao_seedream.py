from typing import List
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.doubao_seedream import DoubaoSeedream
from .modelverse_api.utils import imageurl2tensor


MODELS = [
    "doubao-seedream-5-5-261028",
    "doubao-seedream-5-0-pro-260628-global",
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
]


def _normalize_images(images) -> List:
    if images is None:
        return []
    if isinstance(images, list):
        return images
    if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] > 1:
        return [images[i:i + 1] for i in range(images.shape[0])]
    return [images]


class DoubaoSeedreamNode:
    """
    Doubao Seedream text-to-image and reference-image generation.

    Endpoint: POST /v1/images/generations
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {
                    "default": "doubao-seedream-5-5-261028",
                    "tooltip": "Doubao Seedream model name",
                }),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the image to generate",
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
                "images": (IO.IMAGE, {
                    "tooltip": "Optional reference image(s) for image-to-image",
                    "forceInput": False,
                    "default": None,
                }),
                "size": (IO.STRING, {
                    "default": "2K",
                    "tooltip": "Output size: 1K/1.5K/2K/3K/4K, auto, or <width>x<height>",
                }),
                "output_format": (["", "png", "jpeg"], {
                    "default": "",
                    "tooltip": "Output format (supported by 5.x models); empty uses the model default",
                }),
                "watermark": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "Whether to add the AI generated watermark",
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE/Doubao"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      model,
                      prompt,
                      num_requests=1,
                      images=None,
                      size="2K",
                      output_format="",
                      watermark=True):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")

        image_list = _normalize_images(images)
        if image_list:
            print("INFO:", f"{len(image_list)} reference image(s) included.")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [mv_client.async_send_request(DoubaoSeedream(
            prompt=prompt,
            model=model,
            images=image_list if image_list else None,
            size=size,
            output_format=output_format if output_format else None,
            watermark=watermark,
            response_format="url",
        )) for _ in range(num_requests)]

        results = await mv_client.run_tasks(tasks)

        output_images_list = []
        for data_list in results:
            if not data_list:
                print("WARN:", "No output in current request. Skipping...")
            output_images_list.append(imageurl2tensor(data_list))
        print("INFO:", f"{len(output_images_list)}/{num_requests} request made successfully.")
        return (torch.cat(output_images_list, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse DoubaoSeedreamNode": DoubaoSeedreamNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse DoubaoSeedreamNode": "Modelverse Doubao Seedream",
}

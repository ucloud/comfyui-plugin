from typing import List
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.wan_2_7_image import Wan27Image
from .modelverse_api.utils import imageurl2tensor


MODELS = ["wan2.7-image", "wan2.7-image-pro"]


def _normalize_images(images) -> List:
    if images is None:
        return []
    if isinstance(images, list):
        return images
    if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] > 1:
        return [images[i:i + 1] for i in range(images.shape[0])]
    return [images]


class Wan27ImageNode:
    """
    Wan 2.7 Image text-to-image and image editing.

    Endpoint: POST /v1/images/generations
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {
                    "default": "wan2.7-image",
                    "tooltip": "Wan 2.7 Image model name",
                }),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt. Can be left empty when reference images are provided",
                }),
                "size": (IO.STRING, {
                    "default": "2K",
                    "tooltip": "Output size: 1K/2K/4K or <width>*<height>, e.g. 1280*720",
                }),
                "num_images": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of images to generate (1 to 12)",
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
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "control_after_generate": True,
                    "tooltip": "Random seed; 0 or omitted is treated as unset",
                }),
            },
            "optional": {
                "images": (IO.IMAGE, {
                    "tooltip": "Optional reference image(s) for image editing",
                    "forceInput": False,
                    "default": None,
                }),
                "watermark": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "Whether to add a watermark",
                }),
                "thinking_mode": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "Enable deep thinking mode (text-to-image without image input only)",
                }),
                "enable_sequential": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "Enable sequential multi-image / group image generation",
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE/Wan"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      model,
                      prompt,
                      size="2K",
                      num_images=1,
                      num_requests=1,
                      seed=0,
                      images=None,
                      watermark=False,
                      thinking_mode=True,
                      enable_sequential=False):

        image_list = _normalize_images(images)
        if not prompt and not image_list:
            raise ValueError("Prompt or at least one reference image is required")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [mv_client.async_send_request(Wan27Image(
            prompt=prompt,
            model=model,
            images=image_list if image_list else None,
            size=size,
            num_images=num_images,
            seed=seed,
            watermark=watermark,
            thinking_mode=thinking_mode,
            enable_sequential=enable_sequential,
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
    "Modelverse Wan27ImageNode": Wan27ImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse Wan27ImageNode": "Modelverse Wan 2.7 Image",
}

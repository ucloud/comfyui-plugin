from typing import List
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.qwen_image_3_0 import QwenImage30
from .modelverse_api.utils import imageurl2tensor


MODELS = ["qwen-image-3.0", "qwen-image-3.0-pro"]


def _normalize_images(images) -> List:
    if images is None:
        return []
    if isinstance(images, list):
        return images
    if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] > 1:
        return [images[i:i + 1] for i in range(images.shape[0])]
    return [images]


class QwenImage30Node:
    """
    Qwen Image 3.0 text-to-image and image editing.

    Endpoint: POST /v1/images/generations
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {
                    "default": "qwen-image-3.0",
                    "tooltip": "Qwen Image 3.0 model name",
                }),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt. Can be left empty when reference images are provided",
                }),
                "size": (IO.STRING, {
                    "default": "1024*1024",
                    "tooltip": "Output size in the form <width>*<height>, e.g. 1024*1024",
                }),
                "num_images": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of images to generate (1 to 6)",
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
                    "max": 2147483647,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible results. -1 for random seed",
                }),
            },
            "optional": {
                "images": (IO.IMAGE, {
                    "tooltip": "Optional reference image(s) for image editing (1 to 3)",
                    "forceInput": False,
                    "default": None,
                }),
                "negative_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt",
                }),
                "watermark": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "Whether to add a watermark",
                }),
                "prompt_extend": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "Whether to enable prompt extension",
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE/Qwen"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      model,
                      prompt,
                      size="1024*1024",
                      num_images=1,
                      num_requests=1,
                      seed=-1,
                      images=None,
                      negative_prompt="",
                      watermark=False,
                      prompt_extend=True):

        image_list = _normalize_images(images)
        if not prompt and not image_list:
            raise ValueError("Prompt or at least one reference image is required")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [mv_client.async_send_request(QwenImage30(
            prompt=prompt,
            model=model,
            images=image_list if image_list else None,
            size=size,
            num_images=num_images,
            seed=seed + i,
            negative_prompt=negative_prompt,
            watermark=watermark,
            prompt_extend=prompt_extend,
        )) for i in range(num_requests)]

        results = await mv_client.run_tasks(tasks)

        output_images_list = []
        for data_list in results:
            if not data_list:
                print("WARN:", "No output in current request. Skipping...")
            output_images_list.append(imageurl2tensor(data_list))
        print("INFO:", f"{len(output_images_list)}/{num_requests} request made successfully.")
        return (torch.cat(output_images_list, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse QwenImage30Node": QwenImage30Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse QwenImage30Node": "Modelverse Qwen Image 3.0",
}

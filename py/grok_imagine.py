from typing import List
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.grok_imagine import GrokImagine
from .modelverse_api.utils import imageurl2tensor


MODELS = [
    "grok-imagine-image",
    "grok-imagine-image-quality",
]

ASPECT_RATIOS = [
    "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2",
    "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1", "auto",
]


class GrokImagineNode:
    """
    Grok Imagine image generation.

    Endpoint: POST /v1/images/generations
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {
                    "default": "grok-imagine-image",
                    "tooltip": "Grok Imagine model name",
                }),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the image to generate",
                }),
                "num_images": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of images to generate (1 to 10)",
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
                "size": (["1k", "2k"], {
                    "default": "1k",
                    "tooltip": "Output resolution",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "4:3",
                    "tooltip": "Output aspect ratio",
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE/Grok"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      model,
                      prompt,
                      num_images=1,
                      num_requests=1,
                      size="1k",
                      aspect_ratio="4:3"):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [mv_client.async_send_request(GrokImagine(
            prompt=prompt,
            model=model,
            num_images=num_images,
            size=size,
            aspect_ratio=aspect_ratio,
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
    "Modelverse GrokImagineNode": GrokImagineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse GrokImagineNode": "Modelverse Grok Imagine",
}

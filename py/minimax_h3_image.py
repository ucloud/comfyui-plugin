from typing import List
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.minimax_h3_image import (
    MiniMaxH3ImageGeneration,
    MiniMaxH3ImageEdit,
)
from .modelverse_api.utils import imageurl2tensor


MODEL = "minimax-h3-image"


def _normalize_images(images) -> List:
    if images is None:
        return []
    if isinstance(images, list):
        return images
    if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] > 1:
        return [images[i:i + 1] for i in range(images.shape[0])]
    return [images]


class MiniMaxH3ImageNode:
    """
    MiniMax H3 Image generation and editing.

    Generation: POST /v1/images/generations
    Editing:    POST /v1/images/edits
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the image or the edit",
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
                "size": (IO.STRING, {
                    "default": "auto",
                    "tooltip": "Output size: auto or <width>x<height>",
                }),
            },
            "optional": {
                "images": (IO.IMAGE, {
                    "tooltip": "Optional reference image(s); when provided the node uses /v1/images/edits",
                    "forceInput": False,
                    "default": None,
                }),
                "area": (["1K", "2K", "4K"], {
                    "default": "4K",
                    "tooltip": "Output resolution area",
                }),
                "output_format": (["jpeg", "png", "jpg", "webp"], {
                    "default": "jpeg",
                    "tooltip": "Output image format",
                }),
                "aigc_watermark": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "Whether to add an AIGC watermark",
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE/MiniMax"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      prompt,
                      num_images=1,
                      num_requests=1,
                      size="auto",
                      images=None,
                      area="4K",
                      output_format="jpeg",
                      aigc_watermark=False):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")

        image_list = _normalize_images(images)
        mv_client = ModelverseClient(client["api_key"])

        tasks = []
        for _ in range(num_requests):
            if image_list:
                tasks.append(mv_client.async_send_request(MiniMaxH3ImageEdit(
                    prompt=prompt,
                    images=image_list,
                    model=MODEL,
                    num_images=num_images,
                    size=size,
                    area=area,
                    output_format=output_format,
                    response_format="url",
                    aigc_watermark=aigc_watermark,
                )))
            else:
                tasks.append(mv_client.async_send_request(MiniMaxH3ImageGeneration(
                    prompt=prompt,
                    model=MODEL,
                    num_images=num_images,
                    size=size,
                    area=area,
                    output_format=output_format,
                    response_format="url",
                    aigc_watermark=aigc_watermark,
                )))

        results = await mv_client.run_tasks(tasks)

        output_images_list = []
        for data_list in results:
            if not data_list:
                print("WARN:", "No output in current request. Skipping...")
            output_images_list.append(imageurl2tensor(data_list))
        print("INFO:", f"{len(output_images_list)}/{num_requests} request made successfully.")
        return (torch.cat(output_images_list, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse MiniMaxH3ImageNode": MiniMaxH3ImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse MiniMaxH3ImageNode": "Modelverse MiniMax H3 Image",
}

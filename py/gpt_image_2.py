import base64
from typing import List
import torch
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.gpt_image_2 import GPTImage2Generate, GPTImage2Edit
from .modelverse_api.utils import decode_image, images2tensor


MODELS = [
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-2.6-flare",
    "gpt-image-2.6-sunburst",
]


def _normalize_images(images) -> List:
    if images is None:
        return []
    if isinstance(images, list):
        return images
    if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] > 1:
        return [images[i:i + 1] for i in range(images.shape[0])]
    return [images]


def _decode_b64_items(data_list) -> List[torch.Tensor]:
    images = []
    for item in data_list or []:
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
            images.append(decode_image(base64.b64decode(b64v)))
        except Exception:
            continue
    return images


class GPTImage2Node:
    """
    gpt-image-2 family text-to-image and image editing.

    Text-to-image: POST /v1/images/generations
    Image editing: POST /v1/images/edits (multipart/form-data)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {
                    "default": "gpt-image-2",
                    "tooltip": "gpt-image model name; only the 2.x models accept image[] input",
                }),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the image or the edit",
                }),
                "size": (IO.STRING, {
                    "default": "1024x1024",
                    "tooltip": "Output size: auto or <width>x<height>, e.g. 1024x1024, 1024x1536",
                }),
                "num_images": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of images to generate (n); gpt-image-1.5 supports at most 4",
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
                    "tooltip": "Optional reference image(s); when provided the node uses /v1/images/edits",
                    "forceInput": False,
                    "default": None,
                }),
                "mask": (IO.IMAGE, {
                    "tooltip": "Optional mask image for editing (edits only)",
                    "forceInput": False,
                    "default": None,
                }),
                "quality": (["low", "medium", "high"], {
                    "default": "medium",
                    "tooltip": "Image quality",
                }),
                "output_format": (["png", "jpeg"], {
                    "default": "png",
                    "tooltip": "Output image format",
                }),
                "output_compression": (IO.INT, {
                    "default": 100,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Compression strength 0-100, only used for jpeg",
                }),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE/GPT-Image"
    FUNCTION = "execute"

    async def execute(self,
                      client,
                      model,
                      prompt,
                      size="1024x1024",
                      num_images=1,
                      num_requests=1,
                      images=None,
                      mask=None,
                      quality="medium",
                      output_format="png",
                      output_compression=100):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")

        image_list = _normalize_images(images)
        if mask is not None and not image_list:
            raise ValueError("Mask requires at least one reference image")

        mv_client = ModelverseClient(client["api_key"])

        tasks = []
        for _ in range(num_requests):
            if image_list:
                tasks.append(mv_client.async_send_request(GPTImage2Edit(
                    prompt=prompt,
                    images=image_list,
                    model=model,
                    mask=mask,
                    num_images=num_images,
                    size=size,
                    quality=quality,
                    output_format=output_format,
                    output_compression=output_compression,
                )))
            else:
                tasks.append(mv_client.async_send_request(GPTImage2Generate(
                    prompt=prompt,
                    model=model,
                    num_images=num_images,
                    size=size,
                    quality=quality,
                    output_format=output_format,
                    output_compression=output_compression,
                )))

        results = await mv_client.run_tasks(tasks)

        output_images_list: List[torch.Tensor] = []
        for data_list in results:
            if not data_list:
                print("WARN:", "No output in current request. Skipping...")
                continue
            images_out = _decode_b64_items(data_list)
            if not images_out:
                print("WARN:", "No decodable base64 image found.")
                continue
            output_images_list.append(images2tensor(images_out))

        if not output_images_list:
            return (torch.zeros((1, 3, 1, 1)),)

        print("INFO:", f"{len(output_images_list)}/{num_requests} request made successfully.")
        return (torch.cat(output_images_list, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse GPTImage2Node": GPTImage2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse GPTImage2Node": "Modelverse GPT Image 2",
}

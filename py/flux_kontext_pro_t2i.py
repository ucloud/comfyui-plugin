from .modelverse_api.utils import imageurl2tensor
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.flux_kontext_pro import FluxKontextProT2I
import torch
import asyncio

class FluxKontextProT2INode:
    """
    Flux Image Generator Node (Kontext Pro) for text2image task.

    This node uses Modelverse's Flux model to generate high-quality images.

    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text description of the image to generate"}),
                "aspect_ratio": (["21:9", "16:9", "16:10", "4:3", "1:1", "3:4", "10:16", "9:16", "9:21"], {
                    "default": "1:1",
                    "tooltip": "The aspect ratio of the output image, ranging from \"21:9\" to \"9:21\", default is \"1:1\""                
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of images to generate in a single request (1 to 4)"
                }),
                "num_requests": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of request to make (1 to 10)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible results. -1 for random seed"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Guidance scale for generation (0.0 to 10.0)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)

    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "execute"

    def execute(self,
                client,
                prompt,
                num_images=1,
                num_requests=1,
                aspect_ratio="1:1",
                seed=-1,
                guidance_scale=2.5):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")

        print("INFO:", "Running Flux Kontext Pro text-to-image mode.")

        client = ModelverseClient(client["api_key"])

        tasks = [client.async_send_request(FluxKontextProT2I(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                num_images=num_images,
                seed=seed+i,
                guidance_scale=guidance_scale
            ))
            for i in range(num_requests)]

        image_urls = asyncio.run(client.run_tasks(tasks))

        output_images_list = []
        for image_url in image_urls:
            if not image_url:
                print("WARN:", "No image URLs in the generated result in current request. Skipping...")
            output_images = imageurl2tensor(image_url)
            output_images_list.append(output_images)
        print("INFO:", f"{len(output_images_list)}/{num_requests} request made successfully.")
        return (torch.cat(output_images_list, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse FluxKontextProT2INode": FluxKontextProT2INode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse FluxKontextProT2INode": "Modelverse Flux Kontext Pro text2image"
}
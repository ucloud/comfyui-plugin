from .modelverse_api.utils import imageurl2tensor
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.flux_dev import FluxDev
import torch
import asyncio

class FluxDevNode:
    """
    Flux Image Generator Node (Dev)

    This node uses Modelverse's Flux model to generate high-quality images.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text description of the image to generate"}),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 1536,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Image width (512 to 1536)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 1536,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Image height (512 to 1536)"
                }),
                "strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Strength of the image-to-image transformation (0.01 to 1.0)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible results. -1 for random seed"
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
                "num_inference_steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of inference steps (1 to 50)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Guidance scale for generation (0.0 to 10.0)"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "The image for reference.", 
                    "forceInput": False, 
                    "default": None
                })
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)

    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "execute"

    def execute(self,
                client,
                prompt,
                width=864,
                height=1536,
                strength=0.8,
                seed=-1,
                num_images=1,
                num_requests=1,
                num_inference_steps=28,
                guidance_scale=3.5,
                image=None):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")
        print("INFO:", "Running Flux Dev.")
        
        client = ModelverseClient(client["api_key"])

        tasks = [client.async_send_request(FluxDev(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_images=num_images,
                num_inference_steps=num_inference_steps,
                seed=seed+i,
                width=width,
                height=height,
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
    "Modelverse FluxDevNode": FluxDevNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse FluxDevNode": "Modelverse Flux Dev"
}
from .modelverse_api.utils import imageurl2tensor
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.flux_kontext_max import FluxKontextMax, FluxKontextMaxMulti
import torch
import asyncio

class FluxKontextMaxNode:
    """
    Flux Image Generator Node (Kontext Max)

    This node uses Modelverse's Flux model to generate high-quality images.

    There are two modes available:
        single image as input: single-image editing mode;
        multiple images as input: multi-image editing mode;
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text description of the image to generate"}),
                "images": ("IMAGE,IMAGE_LIST", {
                    "tooltip": "The image(s) to edit from. If not included, use text-to-image mode.", 
                    "forceInput": False, 
                    "default": None
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
                images,
                num_requests=1,
                seed=-1,
                guidance_scale=3.5):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")
        if images is None:
            raise ValueError("Input image(s) is required")

        mode = "single"
        if isinstance(images, list):
            print("INFO:", "Running Flux Kontext Max multi-image edit mode.")
            print("INFO:", f"{len(images)} image included for the multi-image edit.")
            mode = "multi"
        else:
            print("INFO:", "Running Flux Kontext Max single-image edit mode.")

        
        client = ModelverseClient(client["api_key"])

        if mode == "multi":
            tasks = [client.async_send_request(FluxKontextMaxMulti(
                    prompt=prompt,
                    images=images,
                    guidance_scale=guidance_scale,
                    seed=seed+i,
                ))
                for i in range(num_requests)]
        else:
            tasks = [client.async_send_request(FluxKontextMax(
                    prompt=prompt,
                    image=images,
                    guidance_scale=guidance_scale,
                    seed=seed+i,
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
    "Modelverse FluxKontextMaxNode": FluxKontextMaxNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse FluxKontextMaxNode": "Modelverse Flux Kontext Max"
}
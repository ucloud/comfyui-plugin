from .modelverse_api.utils import imageurl2tensor
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.step1x_edit import Step1xEdit
import torch
import asyncio

class Step1xEditNode:
    """
    Step1X Edit Node

    This node uses Modelverse's Step1X Edit API to transform photos with simple instructions.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt to guide the image edit."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The negative prompt to use."}),
                "image": ("IMAGE", {"default": None, "tooltip": "The image to be edited."}),
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
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of inference steps (1 to 50)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "The CFG scale (0.0 to 20.0)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )

    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "execute"

    def execute(self,
                client,
                prompt,
                image,
                negative_prompt,
                num_requests=1,
                seed=-1,
                num_inference_steps=30,
                guidance_scale=4.0):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")

        if image is None:
            raise ValueError("Input image is required")
        print("INFO:", "Running Step1X-Edit.")

        client = ModelverseClient(client["api_key"])

        tasks = [client.async_send_request(Step1xEdit(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
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
    "Modelverse Step1xEditNode": Step1xEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse Step1xEditNode": "Modelverse Step1X Edit"
}
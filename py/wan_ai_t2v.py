import time
import requests
import os
import tempfile
from .modelverse_api.client import ModelverseClient

class Modelverse_WanAIT2V:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "a beautiful flower"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low quality"}),
                "resolution": (["720P", "480P"], {"default": "720P"}),
                "size": (["1280x720", "720x1280", "832x480", "480x832"], {"default": "1280x720"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "UCLOUD_MODELVERSE"

    def generate_video(self, client, prompt, negative_prompt, resolution, size, seed):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set in the client")
            
        mv_client = ModelverseClient(api_key)

        task_input = {"prompt": prompt}
        if negative_prompt:
            task_input["negative_prompt"] = negative_prompt

        parameters = {
            "resolution": resolution,
            "size": size,
            "seed": seed,
            "duration": 5 # Fixed as per documentation
        }

        # 1. Submit the task
        submit_res = mv_client.submit_task("Wan-AI/Wan2.2-T2V", task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res.get('request_id')}")

        # 2. Poll for the result
        video_url = ""
        while True:
            status_res = mv_client.get_task_status(task_id)
            task_status = status_res.get("output", {}).get("task_status")

            if task_status == "Success":
                video_url = status_res.get("output", {}).get("urls", [None])[0]
                if not video_url:
                    raise Exception("Task succeeded but no video URL was returned.")
                break
            elif task_status == "Failure":
                error_message = status_res.get("output", {}).get("error_message", "Unknown error")
                raise Exception(f"Task failed: {error_message}")
            elif task_status in ["Pending", "Running"]:
                print(f"Task {task_id} is {task_status}, waiting...")
                time.sleep(5)  # Wait for 5 seconds before polling again
            else:
                raise Exception(f"Unknown task status: {task_status}")
        
        # 3. Download video and save to temporary file
        response = requests.get(video_url)
        response.raise_for_status()
        
        # Create temporary file with .mp4 extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(response.content)
        temp_file.close()
        
        return (temp_file.name,)


NODE_CLASS_MAPPINGS = {
    "Modelverse_WanAIT2V": Modelverse_WanAIT2V
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_WanAIT2V": "Modelverse Wan-AI T2V"
}


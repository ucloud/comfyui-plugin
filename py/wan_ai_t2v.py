import time
import requests
import os
import tempfile
import io
from .modelverse_api.client import ModelverseClient

# Import VideoFromFile class for ComfyUI video handling
try:
    # Try the newer import path first
    from comfy_extras.nodes_video import VideoFromFile
except ImportError:
    try:
        # Fallback to older import path
        from comfy.model_management import VideoFromFile
    except ImportError:
        # Final fallback - create a simple wrapper
        class VideoFromFile:
            def __init__(self, video_io):
                self.video_io = video_io
                self.video_data = video_io.getvalue() if hasattr(video_io, 'getvalue') else video_io
            
            def get_dimensions(self):
                # Return default dimensions if we can't determine them
                return (1280, 720)  # width, height

class Modelverse_WanAIT2V:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "a beautiful flower","tooltip": "Text prompt of the image to generate"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low quality","tooltip": "Negative prompt of the image to generate"}),
                "resolution": (["720P", "480P"], {"default": "720P"}),
                "size": (["1280x720", "720x1280", "832x480", "480x832"], {"default": "1280x720"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
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
        
        # 3. Download video
        response = requests.get(video_url)
        response.raise_for_status()
        
        # Extract video data
        video_data = response.content
        
        if not video_data:
            raise Exception("No video data was returned")
        
        print("Video generation completed successfully")
        print(f"Downloaded video data size: {len(video_data)} bytes")
        
        # Convert video data to BytesIO object
        video_io = io.BytesIO(video_data)
        
        # Ensure the BytesIO object is at the beginning
        video_io.seek(0)
        
        # Return VideoFromFile object
        try:
            video_obj = VideoFromFile(video_io)
            print("VideoFromFile object created successfully")
            return (video_obj,)
        except Exception as e:
            print(f"Error creating VideoFromFile object: {e}")
            # If VideoFromFile fails, try returning the raw video data
            # This is a fallback that might work with some ComfyUI versions
            return (video_data,)


NODE_CLASS_MAPPINGS = {
    "Modelverse_WanAIT2V": Modelverse_WanAIT2V
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_WanAIT2V": "Modelverse Wan-AI T2V"
}


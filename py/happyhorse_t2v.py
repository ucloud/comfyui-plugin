"""
HappyHorse 1.0 Text2Video
Model: happyhorse-1.0-t2v
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


MODEL = "happyhorse-1.0-t2v"
RESOLUTIONS = ["480P", "720P", "1080P"]
RATIOS = ["16:9", "9:16", "1:1", "3:4", "4:3"]


class HappyHorseText2VideoNode:
    """
    HappyHorse 1.0 Text2Video - text-to-video generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the desired video",
                }),
            },
            "optional": {
                "resolution": (RESOLUTIONS, {
                    "default": "720P",
                    "tooltip": "Output resolution tier",
                }),
                "ratio": (RATIOS, {
                    "default": "16:9",
                    "tooltip": "Aspect ratio",
                }),
                "duration": (IO.INT, {
                    "default": 5, "min": 1, "max": 15, "step": 1,
                    "tooltip": "Video duration in seconds",
                }),
                "seed": (IO.INT, {
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "Random seed (0 for random)",
                }),
                "watermark": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "Add watermark to output video",
                }),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/HappyHorse"

    def generate(
        self,
        client,
        prompt,
        resolution="720P",
        ratio="16:9",
        duration=5,
        seed=0,
        watermark=False,
    ):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required for HappyHorse T2V")

        mv_client = ModelverseClient(api_key)
        task_input = {"prompt": prompt.strip()}
        parameters = {
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "seed": seed,
        }
        if watermark:
            parameters["watermark"] = True

        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"HappyHorse T2V task submitted: {task_id}")
        video_url = self._poll_task(mv_client, task_id)
        return (video_url, task_id)

    def _poll_task(self, mv_client, task_id, max_retries=180):
        for i in range(max_retries):
            status_res = mv_client.get_task_status(task_id)
            task_status = status_res.get("output", {}).get("task_status")

            if task_status == "Success":
                urls = status_res.get("output", {}).get("urls", [])
                if urls:
                    return urls[0]
                raise Exception("Task succeeded but no video URL returned")
            if task_status == "Failure":
                error = status_res.get("output", {}).get("error_message", "Unknown error")
                raise Exception(f"Task failed: {error}")
            if task_status in ["Pending", "Running"]:
                print(f"Task {task_id}: {task_status} ({i + 1}/{max_retries})")
                time.sleep(5)
                continue
            raise Exception(f"Unknown status: {task_status}")

        raise Exception("Task timed out")


NODE_CLASS_MAPPINGS = {
    "HappyHorse_Text2Video": HappyHorseText2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HappyHorse_Text2Video": "Modelverse HappyHorse T2V",
}

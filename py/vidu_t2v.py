"""
Vidu Text2Video - 文生视频模型
Model: viduq2
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


ASPECT_RATIOS = ["16:9", "9:16", "3:4", "4:3", "1:1"]
RESOLUTIONS = ["540p", "720p", "1080p"]


class ViduText2VideoNode:
    """
    Vidu Text2Video - 文生视频
    Model: viduq2
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "A beautiful sunset over the ocean", "tooltip": "文本提示词，最长2000字符"}),
                "duration": (IO.INT, {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "视频时长(秒)"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "长宽比"}),
                "resolution": (RESOLUTIONS, {"default": "720p", "tooltip": "分辨率"}),
            },
            "optional": {
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子，0表示随机"}),
                "guidance_scale": (IO.FLOAT, {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5, "tooltip": "引导系数"}),
                "bgm": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加背景音乐"}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Vidu"

    def generate(self, client, prompt, duration, aspect_ratio, resolution, seed=0, guidance_scale=7.5, bgm=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        mv_client = ModelverseClient(api_key)

        task_input = {"prompt": prompt}
        parameters = {
            "vidu_type": "text2video",
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "bgm": bgm,
        }

        # Submit task
        submit_res = mv_client.submit_task("viduq2", task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Vidu T2V task submitted: {task_id}")

        # Poll for result
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
            elif task_status == "Failure":
                error = status_res.get("output", {}).get("error_message", "Unknown error")
                raise Exception(f"Task failed: {error}")
            elif task_status in ["Pending", "Running"]:
                print(f"Task {task_id}: {task_status} ({i+1}/{max_retries})")
                time.sleep(5)
            else:
                raise Exception(f"Unknown status: {task_status}")

        raise Exception("Task timed out")


NODE_CLASS_MAPPINGS = {
    "Vidu_Text2Video": ViduText2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Vidu_Text2Video": "Vidu Text2Video",
}

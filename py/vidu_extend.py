"""
Vidu Extend - 视频延长模型
Models: viduq2-pro, viduq2-turbo
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODELS = ["viduq2-pro", "viduq2-turbo"]
RESOLUTIONS = ["540p", "720p", "1080p"]


class ViduExtendNode:
    """
    Vidu Extend - 视频延长
    Models: viduq2-pro, viduq2-turbo
    输入视频时长需4秒-1分钟，延长1-7秒
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": "viduq2-turbo", "tooltip": "viduq2-pro: 效果好, viduq2-turbo: 生成快"}),
                "video_url": (IO.STRING, {"default": "", "tooltip": "需要延长的视频URL (4秒-1分钟)"}),
                "duration": (IO.INT, {"default": 5, "min": 1, "max": 7, "step": 1, "tooltip": "延长时长(秒), 1-7"}),
                "resolution": (RESOLUTIONS, {"default": "720p", "tooltip": "分辨率"}),
            },
            "optional": {
                "last_frame_image": (IO.IMAGE, {"tooltip": "延长到尾帧的参考图像"}),
                "last_frame_url": (IO.STRING, {"default": "", "tooltip": "延长到尾帧的参考图像URL"}),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "延长提示词，控制延长内容"}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Vidu"

    def generate(self, client, model, video_url, duration, resolution,
                 last_frame_image=None, last_frame_url="", prompt=""):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        if not video_url or not video_url.strip():
            raise ValueError("必须提供 video_url")

        mv_client = ModelverseClient(api_key)

        task_input = {
            "video_url": video_url.strip(),
        }

        # Optional last frame
        has_last_url = last_frame_url and last_frame_url.strip()
        has_last_image = last_frame_image is not None
        if has_last_url and has_last_image:
            raise ValueError("尾帧：请提供 image 或 url，不能同时提供")

        if has_last_url:
            task_input["last_frame_url"] = last_frame_url.strip()
        elif has_last_image:
            task_input["last_frame_url"] = image_to_base64(last_frame_image)

        if prompt and prompt.strip():
            task_input["prompt"] = prompt.strip()

        parameters = {
            "vidu_type": "extend",
            "duration": duration,
            "resolution": resolution,
        }

        # Submit task
        submit_res = mv_client.submit_task(model, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Vidu Extend task submitted: {task_id}")

        # Poll for result
        video_url_result = self._poll_task(mv_client, task_id)

        return (video_url_result, task_id)

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
    "Vidu_Extend": ViduExtendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Vidu_Extend": "Vidu Extend",
}

"""
Vidu Img2Video - 图生视频模型
Models: viduq2-pro, viduq2-turbo
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODELS = ["viduq2-pro", "viduq2-turbo"]
RESOLUTIONS = ["540p", "720p", "1080p"]
MOVEMENT_AMPLITUDES = ["auto", "small", "medium", "large"]


class ViduImg2VideoNode:
    """
    Vidu Img2Video - 图生视频
    Models: viduq2-pro (效果好，细节丰富), viduq2-turbo (效果好，生成快)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": "viduq2-pro", "tooltip": "viduq2-pro: 效果好细节丰富, viduq2-turbo: 生成快"}),
                "duration": (IO.INT, {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "视频时长(秒)"}),
                "resolution": (RESOLUTIONS, {"default": "720p", "tooltip": "分辨率"}),
                "movement_amplitude": (MOVEMENT_AMPLITUDES, {"default": "auto", "tooltip": "运动幅度"}),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE, {"tooltip": "首帧图片"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "首帧图片URL (与image二选一)"}),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "文本提示词，最长2000字符"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
                "bgm": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加背景音乐"}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Vidu"

    def generate(self, client, model, duration, resolution, movement_amplitude,
                 first_frame_image=None, first_frame_url="", prompt="", seed=0, bgm=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        mv_client = ModelverseClient(api_key)

        # Validate first frame input
        has_url = first_frame_url and first_frame_url.strip()
        has_image = first_frame_image is not None

        if has_url and has_image:
            raise ValueError("请提供 first_frame_image 或 first_frame_url，不能同时提供")
        if not has_url and not has_image:
            raise ValueError("必须提供 first_frame_image 或 first_frame_url")

        task_input = {}
        if has_url:
            task_input["first_frame_url"] = first_frame_url.strip()
        else:
            task_input["first_frame_url"] = image_to_base64(first_frame_image)

        if prompt and prompt.strip():
            task_input["prompt"] = prompt.strip()

        parameters = {
            "vidu_type": "img2video",
            "duration": duration,
            "resolution": resolution,
            "movement_amplitude": movement_amplitude,
            "seed": seed,
            "bgm": bgm,
        }

        # Submit task
        submit_res = mv_client.submit_task(model, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Vidu I2V task submitted: {task_id}")

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
    "Vidu_Img2Video": ViduImg2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Vidu_Img2Video": "Vidu Img2Video",
}

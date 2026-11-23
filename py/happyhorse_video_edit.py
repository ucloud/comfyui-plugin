"""
HappyHorse 1.0 Video Edit - 视频编辑
Model: happyhorse-1.0-video-edit
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


MODEL = "happyhorse-1.0-video-edit"
RESOLUTIONS = ["720P", "1080P"]
AUDIO_SETTINGS = ["auto", "origin"]


class HappyHorseVideoEditNode:
    """HappyHorse 视频编辑。源视频 URL + 编辑指令，可选参考图 URL 列表。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "视频编辑指令"}),
                "video_url": (IO.STRING, {"default": "", "tooltip": "待编辑视频 URL"}),
            },
            "optional": {
                "images": (IO.STRING, {"multiline": True, "default": "", "tooltip": "参考图 URL，每行一个"}),
                "resolution": (RESOLUTIONS, {"default": "1080P", "tooltip": "输出分辨率"}),
                "watermark": (IO.BOOLEAN, {"default": True, "tooltip": "是否添加水印"}),
                "audio_setting": (AUDIO_SETTINGS, {"default": "auto", "tooltip": "auto: 智能重配; origin: 保留原声"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/HappyHorse"

    def generate(self, client, prompt, video_url, images="", resolution="1080P",
                 watermark=True, audio_setting="auto", seed=0):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")
        if not video_url or not video_url.strip():
            raise ValueError("video_url 不能为空")

        task_input = {"prompt": prompt, "video_url": video_url.strip()}
        image_list = [line.strip() for line in (images or "").splitlines() if line.strip()]
        if image_list:
            task_input["images"] = image_list

        parameters = {
            "resolution": resolution,
            "watermark": watermark,
            "audio_setting": audio_setting,
            "seed": seed,
        }

        mv_client = ModelverseClient(api_key)
        print("Submitting HappyHorse Video Edit task")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"HappyHorse Video Edit task submitted: {task_id}")
        video_url_result = self._poll_task(mv_client, task_id)
        return (video_url_result, task_id)

    def _poll_task(self, mv_client, task_id, max_retries=240):
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
    "HappyHorse_VideoEdit": HappyHorseVideoEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HappyHorse_VideoEdit": "Modelverse HappyHorse Video Edit",
}

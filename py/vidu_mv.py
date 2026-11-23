"""
Vidu MV - 图片 + 音频生成 MV
Model: vidu-mv
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODEL = "vidu-mv"
ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4"]


def _to_data_url(image):
    data_url = image_to_base64(image)
    return data_url.replace("data:image/JPEG", "data:image/jpeg").replace("data:image/PNG", "data:image/png")


class ViduMvNode:
    """
    Vidu MV。images 1-7 张，支持 URL 或 Base64 data URI；
    audio_url 支持音频 URL 或 data URI。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "images": (IO.STRING, {"multiline": True, "default": "",
                                       "tooltip": "1-7 张图片 URL，每行一个（也可用下方 IMAGE 输入）"}),
                "audio_url": (IO.STRING, {"default": "", "tooltip": "音频 URL（mp3/wav 等）"}),
            },
            "optional": {
                "image_1": (IO.IMAGE, {"tooltip": "第 1 张图片（与 images 文本合并）"}),
                "image_2": (IO.IMAGE, {"tooltip": "第 2 张图片"}),
                "image_3": (IO.IMAGE, {"tooltip": "第 3 张图片"}),
                "image_4": (IO.IMAGE, {"tooltip": "第 4 张图片"}),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "MV 画面风格提示词"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "输出宽高比"}),
                "resolution": (["540p", "720p", "1080p"], {"default": "720p", "tooltip": "输出分辨率"}),
                "add_subtitle": (IO.BOOLEAN, {"default": False, "tooltip": "是否生成字幕"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Vidu"

    def generate(self, client, images, audio_url, image_1=None, image_2=None,
                 image_3=None, image_4=None, prompt="", aspect_ratio="16:9",
                 resolution="720p", add_subtitle=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        image_list = [line.strip() for line in (images or "").splitlines() if line.strip()]
        for img in (image_1, image_2, image_3, image_4):
            if img is not None:
                image_list.append(_to_data_url(img))

        if not 1 <= len(image_list) <= 7:
            raise ValueError(f"Vidu MV 需要 1-7 张图片，当前 {len(image_list)} 张")
        if not audio_url or not audio_url.strip():
            raise ValueError("audio_url 不能为空")

        task_input = {
            "prompt": prompt,
            "images": image_list,
            "audio_url": audio_url.strip(),
        }
        parameters = {
            "vidu_type": "one-click/mv",
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "add_subtitle": add_subtitle,
        }

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Vidu MV task: images={len(image_list)}")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Vidu MV task submitted: {task_id}")
        video_url = self._poll_task(mv_client, task_id)
        return (video_url, task_id)

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
    "Vidu_MV": ViduMvNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Vidu_MV": "Modelverse Vidu MV",
}

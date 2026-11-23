"""
PixVerse V6 - 文生/首尾帧/参考图/视频延展
Model: pixverse-v6
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODEL = "pixverse-v6"
RESOLUTIONS = ["360p", "540p", "720p", "1080p"]
ASPECT_RATIOS = ["16:9", "4:3", "1:1", "3:4", "9:16", "2:3", "3:2", "21:9"]


def _to_base64(image):
    data_url = image_to_base64(image)
    return data_url.split(",", 1)[1] if "," in data_url else data_url


class PixverseV6Node:
    """
    PixVerse V6。first_frame/last_frame/img_url 支持 URL 或 Base64(data URL)；
    video_url 用于视频延展。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "文本提示词"}),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE, {"tooltip": "首帧图片（与 first_frame_url 二选一）"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "首帧图片 URL"}),
                "last_frame_image": (IO.IMAGE, {"tooltip": "尾帧图片（与 last_frame_url 二选一）"}),
                "last_frame_url": (IO.STRING, {"default": "", "tooltip": "尾帧图片 URL"}),
                "reference_image": (IO.IMAGE, {"tooltip": "参考图（与 img_url 二选一）"}),
                "img_url": (IO.STRING, {"default": "", "tooltip": "参考图 URL"}),
                "video_url": (IO.STRING, {"default": "", "tooltip": "待延展视频 URL"}),
                "resolution": (RESOLUTIONS, {"default": "720p", "tooltip": "输出分辨率"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "输出宽高比"}),
                "duration": (IO.INT, {"default": 5, "min": 1, "max": 15, "step": 1, "tooltip": "时长(秒)，1-15"}),
                "generate_audio": (IO.INT, {"default": 0, "min": 0, "max": 1, "step": 1, "tooltip": "0 关闭 / 1 开启音频"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Pixverse"

    def generate(self, client, prompt, first_frame_image=None, first_frame_url="",
                 last_frame_image=None, last_frame_url="", reference_image=None,
                 img_url="", video_url="", resolution="720p", aspect_ratio="16:9",
                 duration=5, generate_audio=0, seed=0):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        task_input = {"prompt": prompt}
        first = self._image_value(first_frame_image, first_frame_url, "first_frame")
        if first:
            task_input["first_frame_url"] = first
        last = self._image_value(last_frame_image, last_frame_url, "last_frame")
        if last:
            task_input["last_frame_url"] = last
        ref = self._image_value(reference_image, img_url, "reference_image")
        if ref:
            task_input["img_url"] = ref
        if video_url and video_url.strip():
            task_input["video_url"] = video_url.strip()

        parameters = {
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            "seed": seed,
        }

        mv_client = ModelverseClient(api_key)
        print("Submitting PixVerse V6 task")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"PixVerse V6 task submitted: {task_id}")
        video_url_result = self._poll_task(mv_client, task_id)
        return (video_url_result, task_id)

    @staticmethod
    def _image_value(image, url, label):
        has_url = bool(url and url.strip())
        has_image = image is not None
        if has_url and has_image:
            raise ValueError(f"请提供 {label} 图片或 {label} URL，不能同时提供")
        if has_url:
            return url.strip()
        if has_image:
            return _to_base64(image)
        return ""

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
    "Pixverse_V6": PixverseV6Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixverse_V6": "Modelverse Pixverse V6",
}

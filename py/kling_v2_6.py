"""
Kling V2.6 - 文生视频 / 图生视频
Model: kling-v2-6
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODEL = "kling-v2-6"
KLING_TYPES = ["t2v", "i2v"]
MODES = ["pro"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1"]


def _to_raw_base64(image):
    data_url = image_to_base64(image)
    return data_url.split(",", 1)[1] if "," in data_url else data_url


class KlingV26Node:
    """
    Kling V2.6。i2v 时 image / image_tail 支持 URL 或原始 base64（不带 data 前缀）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "生成提示词"}),
                "kling_type": (KLING_TYPES, {"default": "t2v", "tooltip": "t2v: 文生视频; i2v: 图生视频"}),
            },
            "optional": {
                "negative_prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "负向提示词"}),
                "first_frame_image": (IO.IMAGE, {"tooltip": "i2v 首帧图片（与 first_frame_url 二选一）"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "i2v 首帧图片 URL"}),
                "last_frame_image": (IO.IMAGE, {"tooltip": "i2v 尾帧图片（与 last_frame_url 二选一）"}),
                "last_frame_url": (IO.STRING, {"default": "", "tooltip": "i2v 尾帧图片 URL"}),
                "mode": (MODES, {"default": "pro", "tooltip": "v2.6 目前仅支持 pro"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "输出宽高比"}),
                "duration": (IO.INT, {"default": 5, "min": 5, "max": 10, "step": 5, "tooltip": "时长，仅支持 5 或 10 秒"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Kling"

    def generate(self, client, prompt, kling_type, negative_prompt="",
                 first_frame_image=None, first_frame_url="",
                 last_frame_image=None, last_frame_url="",
                 mode="pro", aspect_ratio="16:9", duration=5):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        task_input = {"prompt": prompt}
        if negative_prompt and negative_prompt.strip():
            task_input["negative_prompt"] = negative_prompt.strip()

        if kling_type == "i2v":
            task_input["image"] = self._image_value(first_frame_image, first_frame_url, "first_frame")
            tail = self._image_value(last_frame_image, last_frame_url, "last_frame", required=False)
            if tail:
                task_input["image_tail"] = tail

        parameters = {
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Kling V2.6 task: type={kling_type}")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Kling V2.6 task submitted: {task_id}")
        video_url = self._poll_task(mv_client, task_id)
        return (video_url, task_id)

    @staticmethod
    def _image_value(image, url, label, required=True):
        has_url = bool(url and url.strip())
        has_image = image is not None
        if has_url and has_image:
            raise ValueError(f"请提供 {label}_image 或 {label}_url，不能同时提供")
        if has_url:
            return url.strip()
        if has_image:
            return _to_raw_base64(image)
        if required:
            raise ValueError(f"i2v 模式必须提供 {label}_image 或 {label}_url")
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
    "Kling_V2_6": KlingV26Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kling_V2_6": "Modelverse Kling V2.6",
}

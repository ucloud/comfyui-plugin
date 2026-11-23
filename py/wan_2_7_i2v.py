"""
Wan 2.7 I2V - 图生视频（首帧/尾帧/首段视频）
Model: wan2.7-i2v
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODEL = "wan2.7-i2v"
RESOLUTIONS = ["720P", "1080P"]


def _to_data_url(image):
    data_url = image_to_base64(image)
    return data_url.replace("data:image/JPEG", "data:image/jpeg").replace("data:image/PNG", "data:image/png")


class Wan27I2VNode:
    """Wan 2.7 图生视频。media 中 first_frame/last_frame 支持 URL 或 data URL base64。"""

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
                "first_clip_url": (IO.STRING, {"default": "", "tooltip": "首段视频 URL（MP4/MOV）"}),
                "driving_audio_url": (IO.STRING, {"default": "", "tooltip": "驱动音频 URL"}),
                "resolution": (RESOLUTIONS, {"default": "720P", "tooltip": "输出分辨率"}),
                "duration": (IO.INT, {"default": 5, "min": 5, "max": 10, "step": 5, "tooltip": "时长，仅支持 5 或 10 秒"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 4294967295, "tooltip": "随机种子"}),
                "prompt_extend": (IO.BOOLEAN, {"default": True, "tooltip": "是否启用提示词智能扩写"}),
                "watermark": (IO.BOOLEAN, {"default": True, "tooltip": "是否添加水印"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Wan"

    def generate(self, client, prompt, first_frame_image=None, first_frame_url="",
                 last_frame_image=None, last_frame_url="", first_clip_url="",
                 driving_audio_url="", resolution="720P", duration=5, seed=0,
                 prompt_extend=True, watermark=True):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        media = []
        first = self._media_item("first_frame", first_frame_image, first_frame_url, "first_frame", required=False)
        if first:
            media.append(first)
        last = self._media_item("last_frame", last_frame_image, last_frame_url, "last_frame", required=False)
        if last:
            media.append(last)
        if first_clip_url and first_clip_url.strip():
            media.append({"type": "first_clip", "url": first_clip_url.strip()})
        if driving_audio_url and driving_audio_url.strip():
            media.append({"type": "driving_audio", "url": driving_audio_url.strip()})

        if not any(m["type"] in ("first_frame", "last_frame", "first_clip") for m in media):
            raise ValueError("必须至少提供 first_frame、last_frame 或 first_clip 之一")

        parameters = {
            "resolution": resolution,
            "duration": duration,
            "seed": seed,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
        }

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Wan 2.7 I2V task: media={[m['type'] for m in media]}")
        submit_res = mv_client.submit_task(MODEL, {"prompt": prompt, "media": media}, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Wan 2.7 I2V task submitted: {task_id}")
        video_url = self._poll_task(mv_client, task_id)
        return (video_url, task_id)

    @staticmethod
    def _media_item(media_type, image, url, label, required=True):
        has_url = bool(url and url.strip())
        has_image = image is not None
        if has_url and has_image:
            raise ValueError(f"请提供 {label}_image 或 {label}_url，不能同时提供")
        if has_url:
            return {"type": media_type, "url": url.strip()}
        if has_image:
            return {"type": media_type, "url": _to_data_url(image)}
        if required:
            raise ValueError(f"必须提供 {label}_image 或 {label}_url")
        return None

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
    "Wan27_I2V": Wan27I2VNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan27_I2V": "Modelverse Wan2.7 I2V",
}

"""
Wan 2.7 R2V - 参考生视频
Model: wan2.7-r2v
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODEL = "wan2.7-r2v"
RESOLUTIONS = ["720P", "1080P"]
RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]


def _to_data_url(image):
    data_url = image_to_base64(image)
    return data_url.replace("data:image/JPEG", "data:image/jpeg").replace("data:image/PNG", "data:image/png")


class Wan27R2VNode:
    """Wan 2.7 参考生视频。reference_image / first_frame 支持 URL 或 data URL base64。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "文本提示词"}),
            },
            "optional": {
                "reference_image_url": (IO.STRING, {"default": "", "tooltip": "参考图 URL（可与 reference_image 二选一）"}),
                "reference_image": (IO.IMAGE, {"tooltip": "参考图片"}),
                "reference_video_url": (IO.STRING, {"default": "", "tooltip": "参考视频 URL"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "首帧图片 URL"}),
                "first_frame_image": (IO.IMAGE, {"tooltip": "首帧图片"}),
                "negative_prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "负向提示词"}),
                "resolution": (RESOLUTIONS, {"default": "1080P", "tooltip": "输出分辨率"}),
                "ratio": (RATIOS, {"default": "16:9", "tooltip": "输出宽高比"}),
                "duration": (IO.INT, {"default": 5, "min": 2, "max": 15, "step": 1, "tooltip": "时长(秒)，2-15"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 4294967295, "tooltip": "随机种子"}),
                "prompt_extend": (IO.BOOLEAN, {"default": True, "tooltip": "是否启用提示词智能扩写"}),
                "watermark": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加 AI Generated 水印"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Wan"

    def generate(self, client, prompt, reference_image_url="", reference_image=None,
                 reference_video_url="", first_frame_url="", first_frame_image=None,
                 negative_prompt="", resolution="720P", ratio="16:9", duration=5,
                 seed=0, prompt_extend=True, watermark=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        media = []
        ref_image = self._media_item("reference_image", reference_image, reference_image_url, "reference_image")
        if ref_image:
            media.append(ref_image)
        if reference_video_url and reference_video_url.strip():
            media.append({"type": "reference_video", "url": reference_video_url.strip()})
        first = self._media_item("first_frame", first_frame_image, first_frame_url, "first_frame")
        if first:
            media.append(first)

        if not any(m["type"] in ("reference_image", "reference_video") for m in media):
            raise ValueError("必须至少提供 reference_image 或 reference_video")

        task_input = {"prompt": prompt, "media": media}
        if negative_prompt and negative_prompt.strip():
            task_input["negative_prompt"] = negative_prompt.strip()

        parameters = {
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "seed": seed,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
        }

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Wan 2.7 R2V task: media={[m['type'] for m in media]}")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Wan 2.7 R2V task submitted: {task_id}")
        video_url = self._poll_task(mv_client, task_id)
        return (video_url, task_id)

    @staticmethod
    def _media_item(media_type, image, url, label):
        has_url = bool(url and url.strip())
        has_image = image is not None
        if has_url and has_image:
            raise ValueError(f"请提供 {label}_url 或 {label} 图片，不能同时提供")
        if has_url:
            return {"type": media_type, "url": url.strip()}
        if has_image:
            return {"type": media_type, "url": _to_data_url(image)}
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
    "Wan27_R2V": Wan27R2VNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan27_R2V": "Modelverse Wan2.7 R2V",
}

"""
Wan 3.0 Video - 多模态视频生成（media 自动判定素材类型）
Model: wan3.0-video
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODEL = "wan3.0-video"
RESOLUTIONS = ["480P", "720P", "1080P"]
RATIOS = ["adaptive", "16:9", "9:16", "1:1", "4:3", "3:4"]


def _to_data_url(image):
    data_url = image_to_base64(image)
    return data_url.replace("data:image/JPEG", "data:image/jpeg").replace("data:image/PNG", "data:image/png")


class Wan30VideoNode:
    """
    Wan 3.0 视频生成。根据提供的素材自动组装 media：
    图片走 data URL base64，视频/音频为公网 URL。
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
                "reference_image_url": (IO.STRING, {"default": "", "tooltip": "参考图 URL"}),
                "reference_image": (IO.IMAGE, {"tooltip": "参考图片（与 reference_image_url 二选一）"}),
                "reference_video_url": (IO.STRING, {"default": "", "tooltip": "参考视频 URL"}),
                "reference_audio_url": (IO.STRING, {"default": "", "tooltip": "参考音频 URL"}),
                "resolution": (RESOLUTIONS, {"default": "1080P", "tooltip": "输出分辨率"}),
                "ratio": (RATIOS, {"default": "adaptive", "tooltip": "输出宽高比；adaptive 为自适应"}),
                "duration": (IO.INT, {"default": 5, "min": 2, "max": 30, "step": 1, "tooltip": "时长(秒)，2-30"}),
                "audio": (IO.BOOLEAN, {"default": False, "tooltip": "是否启用音频"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
                "prompt_extend": (IO.BOOLEAN, {"default": False, "tooltip": "是否启用提示词智能扩写"}),
                "watermark": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加水印"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Wan"

    def generate(self, client, prompt, first_frame_image=None, first_frame_url="",
                 last_frame_image=None, last_frame_url="", reference_image_url="",
                 reference_image=None, reference_video_url="", reference_audio_url="",
                 resolution="1080P", ratio="adaptive", duration=5, audio=False,
                 seed=0, prompt_extend=False, watermark=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        media = []
        media.extend(self._media("first_frame", first_frame_image, first_frame_url, "first_frame"))
        media.extend(self._media("last_frame", last_frame_image, last_frame_url, "last_frame"))
        media.extend(self._media("reference_image", reference_image, reference_image_url, "reference_image"))
        if reference_video_url and reference_video_url.strip():
            media.append({"type": "reference_video", "url": reference_video_url.strip()})
        if reference_audio_url and reference_audio_url.strip():
            media.append({"type": "reference_audio", "url": reference_audio_url.strip()})

        parameters = {
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "audio": audio,
            "seed": seed,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
        }

        task_input = {"prompt": prompt}
        if media:
            task_input["media"] = media

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Wan 3.0 task: media={[m['type'] for m in media]}")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Wan 3.0 task submitted: {task_id}")
        video_url = self._poll_task(mv_client, task_id)
        return (video_url, task_id)

    @staticmethod
    def _media(media_type, image, url, label):
        has_url = bool(url and url.strip())
        has_image = image is not None
        if has_url and has_image:
            raise ValueError(f"请提供 {label}_url 或 {label} 图片，不能同时提供")
        if has_url:
            return [{"type": media_type, "url": url.strip()}]
        if has_image:
            return [{"type": media_type, "url": _to_data_url(image)}]
        return []

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
    "Wan30_Video": Wan30VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan30_Video": "Modelverse Wan3.0 Video",
}

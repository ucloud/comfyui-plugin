"""
Kling V4 / V4 Flash - 文生视频与首帧图生视频
Models: kling-v4, kling-v4-flash
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODELS = ["kling-v4", "kling-v4-flash"]
KLING_TYPES = ["t2v", "i2v"]
RESOLUTIONS = ["720p", "1080p"]
ASPECT_RATIOS = ["21:9", "16:9", "9:16", "1:1", "4:3", "3:4", "adaptive"]


def _to_base64(image):
    data_url = image_to_base64(image)
    return data_url.split(",", 1)[1] if "," in data_url else data_url


class KlingV4Node:
    """
    Kling V4/V4 Flash 视频生成。
    t2v: 仅文本；i2v: 文本 + 首帧图（IMAGE 走 base64 contents，URL 走 first_frame_url）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": "kling-v4", "tooltip": "kling-v4: 720p/1080p; kling-v4-flash: 仅 720p"}),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "生成提示词"}),
                "kling_type": (KLING_TYPES, {"default": "t2v", "tooltip": "t2v: 文生视频; i2v: 首帧图生视频"}),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE, {"tooltip": "i2v 首帧图片（与 first_frame_url 二选一）"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "i2v 首帧图片 URL（与 first_frame_image 二选一）"}),
                "duration": (IO.INT, {"default": 5, "min": 3, "max": 30, "step": 1, "tooltip": "时长(秒)，3-30；flash 最大 20"}),
                "resolution": (RESOLUTIONS, {"default": "1080p", "tooltip": "输出分辨率；flash 仅支持 720p"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "仅 t2v 生效；i2v 不要传该参数"}),
                "audio": (IO.BOOLEAN, {"default": False, "tooltip": "false: native 原生音频; true: original 保留素材原声"}),
                "watermark": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加水印"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Kling"

    def generate(self, client, model, prompt, kling_type, first_frame_image=None,
                 first_frame_url="", duration=5, resolution="1080p",
                 aspect_ratio="16:9", audio=False, watermark=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        has_url = bool(first_frame_url and first_frame_url.strip())
        has_image = first_frame_image is not None

        if has_url and has_image:
            raise ValueError("请提供 first_frame_image 或 first_frame_url，不能同时提供")
        if kling_type == "i2v" and not has_url and not has_image:
            raise ValueError("i2v 模式必须提供 first_frame_image 或 first_frame_url")
        if kling_type == "t2v" and (has_url or has_image):
            raise ValueError("t2v 模式不接受首帧图；请将 kling_type 改为 i2v")
        if model == "kling-v4-flash" and resolution != "720p":
            raise ValueError("kling-v4-flash 仅支持 720p 分辨率")

        task_input = {"prompt": prompt}
        parameters = {
            "kling_type": kling_type,
            "duration": duration,
            "resolution": resolution,
            "audio": audio,
        }
        if model == "kling-v4":
            parameters["watermark_enabled"] = watermark

        if kling_type == "t2v":
            parameters["aspect_ratio"] = aspect_ratio
        elif has_url:
            task_input["first_frame_url"] = first_frame_url.strip()
        else:
            data_url = _to_base64(first_frame_image)
            parameters["contents"] = [{"type": "first_frame", "url": data_url}]

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Kling V4 task: model={model}, type={kling_type}")
        submit_res = mv_client.submit_task(model, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Kling V4 task submitted: {task_id}")
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
    "Kling_V4": KlingV4Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kling_V4": "Modelverse Kling V4",
}

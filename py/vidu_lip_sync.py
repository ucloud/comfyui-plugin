"""
Vidu Lip Sync - 视频对口型
Model: vidu-lip-sync
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


MODEL = "vidu-lip-sync"


class ViduLipSyncNode:
    """
    Vidu 对口型。video_url 必填；audio_url 与 text 至少提供一个（audio_url 优先）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "video_url": (IO.STRING, {"default": "", "tooltip": "源视频 URL（mp4/mov/avi，1-600 秒）"}),
            },
            "optional": {
                "audio_url": (IO.STRING, {"default": "", "tooltip": "驱动音频 URL（与 text 二选一，优先使用）"}),
                "text": (IO.STRING, {"multiline": True, "default": "", "tooltip": "文本驱动内容，支持 <#x#> 停顿标记"}),
                "ref_photo_url": (IO.STRING, {"default": "", "tooltip": "人脸参考图 URL（多人视频时指定目标）"}),
                "speed": (IO.FLOAT, {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "语速，仅文本驱动生效"}),
                "voice_id": (IO.STRING, {"default": "", "tooltip": "音色 ID，仅文本驱动生效"}),
                "volume": (IO.INT, {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "音量，0 为正常音量"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Vidu"

    def generate(self, client, video_url, audio_url="", text="", ref_photo_url="",
                 speed=1.0, voice_id="", volume=0):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")
        if not video_url or not video_url.strip():
            raise ValueError("video_url 不能为空")
        if not (audio_url and audio_url.strip()) and not (text and text.strip()):
            raise ValueError("audio_url 与 text 至少提供一个")

        task_input = {"video_url": video_url.strip()}
        if audio_url and audio_url.strip():
            task_input["audio_url"] = audio_url.strip()
        if text and text.strip():
            task_input["text"] = text.strip()
        if ref_photo_url and ref_photo_url.strip():
            task_input["ref_photo_url"] = ref_photo_url.strip()

        parameters = {
            "vidu_type": "lip-sync",
            "speed": speed,
            "volume": volume,
        }
        if voice_id and voice_id.strip():
            parameters["voice_id"] = voice_id.strip()

        mv_client = ModelverseClient(api_key)
        print("Submitting Vidu Lip Sync task")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Vidu Lip Sync task submitted: {task_id}")
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
    "Vidu_LipSync": ViduLipSyncNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Vidu_LipSync": "Modelverse Vidu Lip Sync",
}

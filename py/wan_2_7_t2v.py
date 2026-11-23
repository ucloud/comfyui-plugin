"""
Wan 2.7 T2V - 文生视频
Model: wan2.7-t2v
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


MODEL = "wan2.7-t2v"
RESOLUTIONS = ["720P", "1080P"]
RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]


class Wan27T2VNode:
    """Wan 2.7 文生视频。支持可选音频驱动（audio_url）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "文本提示词，最多 5000 字符"}),
            },
            "optional": {
                "negative_prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "负向提示词，最多 500 字符"}),
                "audio_url": (IO.STRING, {"default": "", "tooltip": "驱动音频 URL（WAV/MP3，2-30 秒）"}),
                "resolution": (RESOLUTIONS, {"default": "1080P", "tooltip": "输出分辨率档位"}),
                "ratio": (RATIOS, {"default": "16:9", "tooltip": "输出宽高比"}),
                "duration": (IO.INT, {"default": 5, "min": 2, "max": 15, "step": 1, "tooltip": "时长(秒)，2-15"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
                "prompt_extend": (IO.BOOLEAN, {"default": True, "tooltip": "是否启用提示词智能扩写"}),
                "watermark": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加 AI Generated 水印"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Wan"

    def generate(self, client, prompt, negative_prompt="", audio_url="",
                 resolution="1080P", ratio="16:9", duration=5, seed=0,
                 prompt_extend=True, watermark=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        task_input = {"prompt": prompt}
        if negative_prompt and negative_prompt.strip():
            task_input["negative_prompt"] = negative_prompt.strip()
        if audio_url and audio_url.strip():
            task_input["audio_url"] = audio_url.strip()

        parameters = {
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "seed": seed,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
        }

        mv_client = ModelverseClient(api_key)
        print("Submitting Wan 2.7 T2V task")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Wan 2.7 T2V task submitted: {task_id}")
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
    "Wan27_T2V": Wan27T2VNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan27_T2V": "Modelverse Wan2.7 T2V",
}

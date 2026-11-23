"""
Wan 2.7 Video Edit - 视频编辑
Model: wan2.7-videoedit
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


MODEL = "wan2.7-videoedit"
RESOLUTIONS = ["720P", "1080P"]
RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
AUDIO_SETTINGS = ["auto", "origin"]


class Wan27VideoEditNode:
    """Wan 2.7 视频编辑。输入源视频 + 编辑指令，可选参考图 URL。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "视频编辑指令"}),
                "video_url": (IO.STRING, {"default": "", "tooltip": "待编辑视频 URL（MP4/MOV，2-10 秒）"}),
            },
            "optional": {
                "reference_image_url": (IO.STRING, {"default": "", "tooltip": "参考图 URL（最多 4 张，可用逗号或换行分隔）"}),
                "resolution": (RESOLUTIONS, {"default": "1080P", "tooltip": "输出分辨率"}),
                "ratio": (RATIOS, {"default": "16:9", "tooltip": "输出宽高比"}),
                "duration": (IO.INT, {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "输出时长(秒)，0 表示跟随输入"}),
                "audio_setting": (AUDIO_SETTINGS, {"default": "auto", "tooltip": "auto: 智能重配; origin: 保留原声"}),
                "prompt_extend": (IO.BOOLEAN, {"default": True, "tooltip": "是否启用提示词智能扩写"}),
                "watermark": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加 AI Generated 水印"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 4294967295, "tooltip": "随机种子"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Wan"

    def generate(self, client, prompt, video_url, reference_image_url="",
                 resolution="1080P", ratio="16:9",
                 duration=0, audio_setting="auto", prompt_extend=True,
                 watermark=False, seed=0):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")
        if not video_url or not video_url.strip():
            raise ValueError("video_url 不能为空")

        media = [{"type": "video", "url": video_url.strip()}]
        ref_urls = [u.strip() for u in reference_image_url.replace("\n", ",").split(",") if u.strip()]
        if len(ref_urls) > 4:
            raise ValueError("reference_image_url 最多 4 张参考图")
        for url in ref_urls:
            media.append({"type": "reference_image", "url": url})

        parameters = {
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "audio_setting": audio_setting,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
            "seed": seed,
        }

        mv_client = ModelverseClient(api_key)
        print("Submitting Wan 2.7 Video Edit task")
        submit_res = mv_client.submit_task(MODEL, {"prompt": prompt, "media": media}, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Wan 2.7 Video Edit task submitted: {task_id}")
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
    "Wan27_VideoEdit": Wan27VideoEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan27_VideoEdit": "Modelverse Wan2.7 Video Edit",
}

"""
Grok Imagine Video - 文生/图生/参考视频生成
Model: grok-imagine-video
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


MODEL = "grok-imagine-video"
RESOLUTIONS = ["480p", "720p"]
ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]


class GrokImagineVideoNode:
    """
    Grok Imagine Video。img_url 为图生视频输入，reference_urls 为参考图列表（二者互斥）。
    所有图片/视频均为 URL。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "文本提示词"}),
                "resolution": (RESOLUTIONS, {"default": "480p", "tooltip": "输出分辨率"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "输出宽高比"}),
                "duration": (IO.INT, {"default": 8, "min": 1, "max": 15, "step": 1, "tooltip": "时长(秒)，1-15"}),
            },
            "optional": {
                "img_url": (IO.STRING, {"default": "", "tooltip": "图生视频的首帧图片 URL"}),
                "reference_urls": (IO.STRING, {"multiline": True, "default": "",
                                               "tooltip": "参考图/视频 URL，每行一个（与 img_url 互斥）"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Grok"

    def generate(self, client, prompt, resolution="480p", aspect_ratio="16:9",
                 duration=8, img_url="", reference_urls=""):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        has_img = bool(img_url and img_url.strip())
        refs = [line.strip() for line in (reference_urls or "").splitlines() if line.strip()]
        if has_img and refs:
            raise ValueError("img_url 与 reference_urls 不能同时使用")

        task_input = {"prompt": prompt}
        if has_img:
            task_input["img_url"] = img_url.strip()
        if refs:
            task_input["reference_urls"] = refs

        parameters = {
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }

        mv_client = ModelverseClient(api_key)
        print("Submitting Grok Imagine Video task")
        submit_res = mv_client.submit_task(MODEL, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Grok Imagine Video task submitted: {task_id}")
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
    "Grok_Imagine_Video": GrokImagineVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grok_Imagine_Video": "Modelverse Grok Imagine Video",
}

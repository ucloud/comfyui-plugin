"""
Vidu Reference2Video - 参考生视频模型
Model: viduq2
支持1-7张参考图片，生成具备主体一致的视频
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


ASPECT_RATIOS = ["16:9", "9:16", "3:4", "4:3", "1:1"]
RESOLUTIONS = ["540p", "720p", "1080p"]


class ViduReference2VideoNode:
    """
    Vidu Reference2Video - 参考生视频
    Model: viduq2
    支持1-7张参考图片，生成具备主体一致的视频
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "make it dance", "tooltip": "文本提示词，最长2000字符"}),
                "duration": (IO.INT, {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "视频时长(秒)"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "长宽比"}),
                "resolution": (RESOLUTIONS, {"default": "720p", "tooltip": "分辨率"}),
            },
            "optional": {
                "image1": (IO.IMAGE, {"tooltip": "参考图片1"}),
                "image2": (IO.IMAGE, {"tooltip": "参考图片2"}),
                "image3": (IO.IMAGE, {"tooltip": "参考图片3"}),
                "image4": (IO.IMAGE, {"tooltip": "参考图片4"}),
                "image5": (IO.IMAGE, {"tooltip": "参考图片5"}),
                "image6": (IO.IMAGE, {"tooltip": "参考图片6"}),
                "image7": (IO.IMAGE, {"tooltip": "参考图片7"}),
                "image_urls": (IO.STRING, {"default": "", "multiline": True, "tooltip": "参考图片URL列表，每行一个"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
                "bgm": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加背景音乐"}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Vidu"

    def generate(self, client, prompt, duration, aspect_ratio, resolution,
                 image1=None, image2=None, image3=None, image4=None,
                 image5=None, image6=None, image7=None,
                 image_urls="", seed=0, bgm=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        mv_client = ModelverseClient(api_key)

        # Collect images
        images = []
        for img in [image1, image2, image3, image4, image5, image6, image7]:
            if img is not None:
                images.append(image_to_base64(img))

        # Add URL images
        if image_urls and image_urls.strip():
            for url in image_urls.strip().split('\n'):
                url = url.strip()
                if url:
                    images.append(url)

        if not images:
            raise ValueError("至少需要提供1张参考图片")
        if len(images) > 7:
            print(f"Warning: 最多支持7张参考图片，当前{len(images)}张，将只使用前7张")
            images = images[:7]

        task_input = {
            "images": images,
            "prompt": prompt,
        }

        parameters = {
            "vidu_type": "reference2video",
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "seed": seed,
            "bgm": bgm,
        }

        # Submit task
        submit_res = mv_client.submit_task("viduq2", task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Vidu Ref2V task submitted: {task_id}")

        # Poll for result
        video_url = self._poll_task(mv_client, task_id)

        return (video_url, task_id)

    def _poll_task(self, mv_client, task_id, max_retries=180):
        for i in range(max_retries):
            status_res = mv_client.get_task_status(task_id)
            task_status = status_res.get("output", {}).get("task_status")

            if task_status == "Success":
                urls = status_res.get("output", {}).get("urls", [])
                if urls:
                    return urls[0]
                raise Exception("Task succeeded but no video URL returned")
            elif task_status == "Failure":
                error = status_res.get("output", {}).get("error_message", "Unknown error")
                raise Exception(f"Task failed: {error}")
            elif task_status in ["Pending", "Running"]:
                print(f"Task {task_id}: {task_status} ({i+1}/{max_retries})")
                time.sleep(5)
            else:
                raise Exception(f"Unknown status: {task_status}")

        raise Exception("Task timed out")


NODE_CLASS_MAPPINGS = {
    "Vidu_Reference2Video": ViduReference2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Vidu_Reference2Video": "Vidu Reference2Video",
}

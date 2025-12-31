"""
OpenAI Sora2 Img2Video - 图生视频模型
Models: openai/sora-2/image-to-video, openai/sora-2/image-to-video-pro
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODELS = ["openai/sora-2/image-to-video", "openai/sora-2/image-to-video-pro"]
RESOLUTIONS_PRO = ["720p", "1080p"]
DURATIONS = [4, 8, 12]


class SoraImg2VideoNode:
    """
    OpenAI Sora2 Img2Video - 图生视频
    Models: image-to-video (普通版), image-to-video-pro (Pro版，支持1080p)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": "openai/sora-2/image-to-video", "tooltip": "image-to-video: 普通版, image-to-video-pro: Pro版(支持1080p)"}),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE, {"tooltip": "首帧图片"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "首帧图片URL (与image二选一)"}),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "提示词，用于指导视频生成"}),
                "resolution": (RESOLUTIONS_PRO, {"default": "720p", "tooltip": "分辨率 (Pro版支持1080p)"}),
                "duration": (DURATIONS, {"default": 4, "tooltip": "视频时长(秒): 4, 8, 12"}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Sora"

    def generate(self, client, model, first_frame_image=None, first_frame_url="", 
                 prompt="", resolution="720p", duration=4):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        # Validate resolution for non-pro model
        if model == "openai/sora-2/image-to-video" and resolution == "1080p":
            raise ValueError("普通版不支持 1080p，请使用 Pro 版")

        mv_client = ModelverseClient(api_key)

        # Validate first frame input
        has_url = first_frame_url and first_frame_url.strip()
        has_image = first_frame_image is not None

        if has_url and has_image:
            raise ValueError("请提供 first_frame_image 或 first_frame_url，不能同时提供")
        if not has_url and not has_image:
            raise ValueError("必须提供 first_frame_image 或 first_frame_url")

        task_input = {}
        if has_url:
            task_input["first_frame_url"] = first_frame_url.strip()
        else:
            task_input["first_frame_url"] = image_to_base64(first_frame_image)

        if prompt and prompt.strip():
            task_input["prompt"] = prompt.strip()

        parameters = {
            "duration": duration,
        }
        
        # Pro版支持resolution参数
        if model == "openai/sora-2/image-to-video-pro":
            parameters["resolution"] = resolution

        # Submit task
        submit_res = mv_client.submit_task(model, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Sora I2V task submitted: {task_id}")

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
    "Sora_Img2Video": SoraImg2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sora_Img2Video": "Sora Img2Video",
}

"""
OpenAI Sora2 Text2Video - 文生视频模型
Models: openai/sora-2/text-to-video, openai/sora-2/text-to-video-pro
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


MODELS = ["openai/sora-2/text-to-video", "openai/sora-2/text-to-video-pro"]
SIZES = ["720x1280", "1280x720"]
SIZES_PRO = ["720x1280", "1280x720", "1024x1792", "1792x1024"]
DURATIONS = [4, 8, 12]


class SoraText2VideoNode:
    """
    OpenAI Sora2 Text2Video - 文生视频
    Models: text-to-video (普通版), text-to-video-pro (Pro版，支持更多分辨率)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": "openai/sora-2/text-to-video", "tooltip": "text-to-video: 普通版, text-to-video-pro: Pro版"}),
                "prompt": (IO.STRING, {"multiline": True, "default": "A beautiful girl is dancing", "tooltip": "提示词，用于指导视频生成"}),
            },
            "optional": {
                "size": (SIZES_PRO, {"default": "720x1280", "tooltip": "视频尺寸 (Pro版支持更多选项)"}),
                "duration": (DURATIONS, {"default": 4, "tooltip": "视频时长(秒): 4, 8, 12"}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Sora"

    def generate(self, client, model, prompt, size="720x1280", duration=4):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        # Validate size for non-pro model
        if model == "openai/sora-2/text-to-video" and size not in SIZES:
            raise ValueError(f"普通版仅支持 {SIZES}，请使用 Pro 版获取更多分辨率选项")

        mv_client = ModelverseClient(api_key)

        task_input = {"prompt": prompt}
        parameters = {
            "size": size,
            "duration": duration,
        }

        # Submit task
        submit_res = mv_client.submit_task(model, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Sora T2V task submitted: {task_id}")

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
    "Sora_Text2Video": SoraText2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sora_Text2Video": "Sora Text2Video",
}

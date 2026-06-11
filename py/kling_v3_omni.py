"""
Kling V3 Omni - Multimodal video generation and editing model
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.kling_common import (
    ASPECT_RATIOS,
    MODEL_KLING_V3_OMNI,
    MODES,
    REFER_TYPES,
    SOUNDS,
    YES_NO,
)
from .modelverse_api.requests.kling_v3_omni import KlingV3Omni
from comfy.comfy_types.node_typing import IO


class KlingV3OmniNode:
    """
    Kling V3 Omni multimodal video generation.
    Supports reference images, videos, and element library IDs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the desired video",
                }),
            },
            "optional": {
                "negative_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt to avoid unwanted content",
                }),
                "first_frame_image": (IO.IMAGE, {"tooltip": "First frame reference image"}),
                "first_frame_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "First frame URL (use either this OR first_frame_image, not both)",
                }),
                "last_frame_image": (IO.IMAGE, {"tooltip": "End frame reference image"}),
                "last_frame_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "End frame URL (use either this OR last_frame_image, not both)",
                }),
                "reference_video_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "Reference video URL for editing or style reference",
                }),
                "refer_type": (REFER_TYPES, {
                    "default": "feature",
                    "tooltip": "feature: style/motion reference; base: video editing",
                }),
                "keep_original_sound": (YES_NO, {
                    "default": "no",
                    "tooltip": "Keep original audio from reference video",
                }),
                "element_id": (IO.INT, {
                    "default": 0, "min": 0,
                    "tooltip": "Kling element library ID (0 to skip)",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "16:9",
                    "tooltip": "Output aspect ratio (required without first-frame or video editing)",
                }),
                "duration": (IO.INT, {
                    "default": 5, "min": 3, "max": 15, "step": 1,
                    "tooltip": "Video duration in seconds (3–15)",
                }),
                "mode": (MODES, {"default": "std", "tooltip": "std: 720p, pro: 1080p"}),
                "sound": (SOUNDS, {
                    "default": "off",
                    "tooltip": "Generate synchronized audio (must be off when reference_video_url is set)",
                }),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Kling"

    def generate(
        self,
        client,
        prompt,
        negative_prompt="",
        first_frame_image=None,
        first_frame_url="",
        last_frame_image=None,
        last_frame_url="",
        reference_video_url="",
        refer_type="feature",
        keep_original_sound="no",
        element_id=0,
        aspect_ratio="16:9",
        duration=5,
        mode="std",
        sound="off",
    ):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        request = KlingV3Omni(
            prompt=prompt,
            negative_prompt=negative_prompt,
            first_frame=first_frame_image,
            first_frame_url=first_frame_url,
            last_frame=last_frame_image,
            last_frame_url=last_frame_url,
            reference_video_url=reference_video_url,
            refer_type=refer_type,
            keep_original_sound=keep_original_sound,
            element_id=element_id,
            aspect_ratio=aspect_ratio,
            duration=duration,
            mode=mode,
            sound=sound,
        )

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Kling V3 Omni task: model={MODEL_KLING_V3_OMNI}, prompt={prompt!r}")
        submit_res = mv_client.submit_task_request(request)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Kling V3 Omni task submitted: {task_id}")
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
    "Kling_V3_Omni": KlingV3OmniNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kling_V3_Omni": "Modelverse Kling V3 Omni",
}

"""
Kling V3 - Unified text/image-to-video and motion control model
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.kling_common import (
    ASPECT_RATIOS,
    CHARACTER_ORIENTATIONS,
    KLING_V3_TYPES,
    MODEL_KLING_V3,
    MODES,
    SOUNDS,
    YES_NO,
)
from .modelverse_api.requests.kling_v3 import KlingV3
from comfy.comfy_types.node_typing import IO


class KlingV3Node:
    """
    Kling V3 unified video generation.
    Auto-routes to text-to-video, image-to-video, or motion control.
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
                "kling_v3_type": (KLING_V3_TYPES, {
                    "default": "auto",
                    "tooltip": "Task type; auto infers from inputs (image→i2v, video→motion_control)",
                }),
                "first_frame_image": (IO.IMAGE, {"tooltip": "First frame image (image-to-video / motion control)"}),
                "first_frame_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "First frame URL (use either this OR first_frame_image, not both)",
                }),
                "last_frame_image": (IO.IMAGE, {"tooltip": "Last frame image (image-to-video)"}),
                "last_frame_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "Last frame URL (use either this OR last_frame_image, not both)",
                }),
                "reference_video_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "Reference video URL (motion control; auto-selects motion_control when set)",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "Output aspect ratio"}),
                "duration": (IO.INT, {
                    "default": 5, "min": 3, "max": 15, "step": 1,
                    "tooltip": "Video duration in seconds (3–15)",
                }),
                "mode": (MODES, {"default": "std", "tooltip": "std: 720p, pro: 1080p"}),
                "sound": (SOUNDS, {"default": "off", "tooltip": "Generate synchronized audio"}),
                "shot_type": (IO.STRING, {
                    "default": "",
                    "tooltip": "Optional shot type, e.g. 'multi' for multi-shot",
                }),
                "character_orientation": (CHARACTER_ORIENTATIONS, {
                    "default": "image",
                    "tooltip": "Motion control: follow image or video orientation",
                }),
                "keep_original_sound": (YES_NO, {
                    "default": "no",
                    "tooltip": "Motion control: keep original reference video sound",
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
        kling_v3_type="auto",
        first_frame_image=None,
        first_frame_url="",
        last_frame_image=None,
        last_frame_url="",
        reference_video_url="",
        aspect_ratio="16:9",
        duration=5,
        mode="std",
        sound="off",
        shot_type="",
        character_orientation="image",
        keep_original_sound="no",
    ):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        request = KlingV3(
            prompt=prompt,
            negative_prompt=negative_prompt,
            kling_v3_type=kling_v3_type,
            first_frame=first_frame_image,
            first_frame_url=first_frame_url,
            last_frame=last_frame_image,
            last_frame_url=last_frame_url,
            reference_video_url=reference_video_url,
            aspect_ratio=aspect_ratio,
            duration=duration,
            mode=mode,
            sound=sound,
            shot_type=shot_type,
            character_orientation=character_orientation,
            keep_original_sound=keep_original_sound,
        )

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Kling V3 task: model={MODEL_KLING_V3}, type={kling_v3_type}, prompt={prompt!r}")
        submit_res = mv_client.submit_task_request(request)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Kling V3 task submitted: {task_id}")
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
    "Kling_V3": KlingV3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kling_V3": "Modelverse Kling V3",
}

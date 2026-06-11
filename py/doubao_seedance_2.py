"""
Doubao Seedance 2.0 - Text/image-to-video model
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.doubao_seedance_2 import (
    DoubaoSeedance2,
    MODEL,
    RESOLUTIONS,
    RATIOS,
)
from comfy.comfy_types.node_typing import IO


class DoubaoSeedance2Node:
    """
    Doubao Seedance 2.0 text/image-to-video generation.
    Supports text, first/last frame images, reference image/video/audio inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the desired video (recommended: up to 500 characters)",
                }),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE, {"tooltip": "First frame image"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "First frame image URL (use either this OR first_frame_image, not both)"}),
                "last_frame_image": (IO.IMAGE, {"tooltip": "Last frame image"}),
                "last_frame_url": (IO.STRING, {"default": "", "tooltip": "Last frame image URL (use either this OR last_frame_image, not both)"}),
                "reference_image": (IO.IMAGE, {"tooltip": "Reference image"}),
                "reference_image_url": (IO.STRING, {"default": "", "tooltip": "Reference image URL (use either this OR reference_image, not both)"}),
                "reference_video_url": (IO.STRING, {"default": "", "tooltip": "Reference video URL"}),
                "reference_audio_url": (IO.STRING, {"default": "", "tooltip": "Reference audio URL"}),
                "duration": (IO.INT, {"default": 5, "min": 4, "max": 15, "step": 1, "tooltip": "Video duration in seconds (4–15)"}),
                "resolution": (RESOLUTIONS, {"default": "720p", "tooltip": "Output resolution; draft mode supports 480p only"}),
                "ratio": (RATIOS, {"default": "adaptive", "tooltip": "Aspect ratio; adaptive auto-selects the best fit"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "Random seed for reproducible results"}),
                "generate_audio": (IO.BOOLEAN, {"default": False, "tooltip": "Generate audio synchronized with the video"}),
                "camera_fixed": (IO.BOOLEAN, {"default": False, "tooltip": "Fix camera position (no camera movement)"}),
                "watermark": (IO.BOOLEAN, {"default": False, "tooltip": "Add watermark to the output video"}),
                "draft": (IO.BOOLEAN, {"default": False, "tooltip": "Draft/preview mode (480p only)"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Seedance"

    def generate(
        self,
        client,
        prompt,
        first_frame_image=None,
        first_frame_url="",
        last_frame_image=None,
        last_frame_url="",
        reference_image=None,
        reference_image_url="",
        reference_video_url="",
        reference_audio_url="",
        duration=5,
        resolution="720p",
        ratio="adaptive",
        seed=0,
        generate_audio=False,
        camera_fixed=False,
        watermark=False,
        draft=False,
    ):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        request = DoubaoSeedance2(
            prompt=prompt,
            first_frame=first_frame_image,
            first_frame_url=first_frame_url,
            last_frame=last_frame_image,
            last_frame_url=last_frame_url,
            reference_image=reference_image,
            reference_image_url=reference_image_url,
            reference_video_url=reference_video_url,
            reference_audio_url=reference_audio_url,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            seed=seed,
            generate_audio=generate_audio,
            camera_fixed=camera_fixed,
            watermark=watermark,
            draft=draft,
        )

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Seedance 2.0 task: model={MODEL}, prompt={prompt!r}")
        submit_res = mv_client.submit_task_request(request)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Seedance 2.0 task submitted: {task_id}")
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
    "Doubao_Seedance_2": DoubaoSeedance2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Doubao_Seedance_2": "Modelverse Doubao Seedance 2.0",
}

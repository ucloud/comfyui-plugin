"""
Google Veo 3.1 video generation
Models: veo-3.1-generate-001, veo-3.1-fast-generate-001
"""
import base64
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import decode_image, encode_image, fetch_image, tensor2images
from comfy.comfy_types.node_typing import IO


MODELS = ["veo-3.1-generate-001", "veo-3.1-fast-generate-001"]
ASPECT_RATIOS = ["16:9", "9:16"]
RESOLUTIONS = ["720p", "1080p"]
DURATIONS = [4, 6, 8]
PERSON_GENERATIONS = ["dont_allow", "allow_adult"]


def _bytes_to_veo_image(data_bytes, fmt):
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return {
        "bytesBase64Encoded": base64.b64encode(data_bytes).decode("utf-8"),
        "mimeType": mime,
    }


def _tensor_to_veo_image(tensor):
    data_bytes, fmt = encode_image(tensor2images(tensor)[0])
    return _bytes_to_veo_image(data_bytes, fmt)


def _url_to_veo_image(url, label):
    url = url.strip()
    if url.startswith("data:image/") and "," in url:
        header, data = url.split(",", 1)
        mime = header.split(";")[0].replace("data:", "")
        return {"bytesBase64Encoded": data, "mimeType": mime}
    if url.startswith(("http://", "https://")):
        image_data = fetch_image(url)
        img = decode_image(image_data)
        data_bytes, fmt = encode_image(img)
        return _bytes_to_veo_image(data_bytes, fmt)
    raise ValueError(f"{label}: URL must be http(s) or a data:image/...;base64,... value")


def _resolve_veo_image(image, url, label):
    has_url = url and url.strip()
    has_image = image is not None
    if has_url and has_image:
        raise ValueError(f"{label}: provide either image or url, not both")
    if has_url:
        return _url_to_veo_image(url, label)
    if has_image:
        return _tensor_to_veo_image(image)
    return None


class Veo31VideoNode:
    """
    Veo 3.1 video generation.
    Supports text-to-video, image-to-video, and first/last-frame video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {
                    "default": "veo-3.1-generate-001",
                    "tooltip": "veo-3.1-generate-001: standard quality, veo-3.1-fast-generate-001: faster generation",
                }),
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt describing the desired video",
                }),
                "generate_audio": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "Whether to generate synchronized audio (required by Veo API)",
                }),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE, {"tooltip": "First frame image for image-to-video"}),
                "first_frame_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "First frame image URL (use either this OR first_frame_image)",
                }),
                "last_frame_image": (IO.IMAGE, {"tooltip": "Last frame image for start-end video"}),
                "last_frame_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "Last frame image URL (use either this OR last_frame_image)",
                }),
                "negative_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt for unwanted elements",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "16:9",
                    "tooltip": "Output aspect ratio",
                }),
                "resolution": (RESOLUTIONS, {
                    "default": "720p",
                    "tooltip": "Output resolution",
                }),
                "duration": (DURATIONS, {
                    "default": 8,
                    "tooltip": "Video duration in seconds: 4, 6, or 8",
                }),
                "seed": (IO.INT, {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": "Random seed (0 to skip)",
                }),
                "person_generation": (PERSON_GENERATIONS, {
                    "default": "allow_adult",
                    "tooltip": "Safety setting for person/face generation",
                }),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Veo"

    def generate(
        self,
        client,
        model,
        prompt,
        generate_audio=True,
        first_frame_image=None,
        first_frame_url="",
        last_frame_image=None,
        last_frame_url="",
        negative_prompt="",
        aspect_ratio="16:9",
        resolution="720p",
        duration=8,
        seed=0,
        person_generation="allow_adult",
    ):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required for Veo 3.1")

        first_image = _resolve_veo_image(first_frame_image, first_frame_url, "First frame")
        last_image = _resolve_veo_image(last_frame_image, last_frame_url, "Last frame")
        if last_image and not first_image:
            raise ValueError("First frame is required when last frame is provided")

        mv_client = ModelverseClient(api_key)
        task_input = {"prompt": prompt.strip()}
        if negative_prompt and negative_prompt.strip():
            task_input["negative_prompt"] = negative_prompt.strip()
        if first_image:
            task_input["image"] = first_image
        if last_image:
            task_input["last_frame"] = last_image

        parameters = {
            "generate_audio": generate_audio,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "person_generation": person_generation,
        }
        if seed > 0:
            parameters["seed"] = seed

        submit_res = mv_client.submit_task(model, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Veo 3.1 task submitted: model={model}, task_id={task_id}")
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
    "Veo_3_1_Video": Veo31VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo_3_1_Video": "Modelverse Veo 3.1 Video",
}

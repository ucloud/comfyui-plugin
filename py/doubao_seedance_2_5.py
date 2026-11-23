"""
Doubao Seedance 2.5 - 文生/图生/参考生视频
Models: doubao-seedance-2-5-260628, doubao-seedance-2-5-260628-global
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODELS = ["doubao-seedance-2-5-260628", "doubao-seedance-2-5-260628-global"]
RESOLUTIONS = ["480p", "720p", "1080p", "4k"]
RATIOS = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]


def _to_data_url(image):
    data_url = image_to_base64(image)
    # 文档要求 data URL 中的 format 使用小写字母
    return data_url.replace("data:image/JPEG", "data:image/jpeg").replace("data:image/PNG", "data:image/png")


class DoubaoSeedance25Node:
    """
    Seedance 2.5。根据素材自动组合 content：
    text + image_url(first_frame / last_frame / reference_image) +
    video_url(reference_video) + audio_url(reference_audio)。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": MODELS[0], "tooltip": "global 为海外节点，分辨率最高 1080p"}),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "文本提示词"}),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE, {"tooltip": "首帧图片（与 first_frame_url 二选一）"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "首帧图片 URL"}),
                "last_frame_image": (IO.IMAGE, {"tooltip": "尾帧图片（与 last_frame_url 二选一）"}),
                "last_frame_url": (IO.STRING, {"default": "", "tooltip": "尾帧图片 URL"}),
                "reference_image_url": (IO.STRING, {"default": "", "tooltip": "参考图 URL"}),
                "reference_image": (IO.IMAGE, {"tooltip": "参考图片（与 reference_image_url 二选一）"}),
                "reference_video_url": (IO.STRING, {"default": "", "tooltip": "参考视频 URL"}),
                "reference_audio_url": (IO.STRING, {"default": "", "tooltip": "参考音频 URL"}),
                "resolution": (RESOLUTIONS, {"default": "720p", "tooltip": "输出分辨率；global 最高 1080p"}),
                "ratio": (RATIOS, {"default": "adaptive", "tooltip": "输出宽高比"}),
                "duration": (IO.INT, {"default": 5, "min": 4, "max": 30, "step": 1, "tooltip": "时长(秒)，4-30"}),
                "generate_audio": (IO.BOOLEAN, {"default": False, "tooltip": "是否生成同步音频"}),
                "camera_fixed": (IO.BOOLEAN, {"default": False, "tooltip": "是否固定镜头"}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子"}),
                "watermark": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加水印"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Doubao"

    def generate(self, client, model, prompt, first_frame_image=None, first_frame_url="",
                 last_frame_image=None, last_frame_url="", reference_image_url="",
                 reference_image=None, reference_video_url="", reference_audio_url="",
                 resolution="720p", ratio="adaptive", duration=5,
                 generate_audio=False, camera_fixed=False, seed=0, watermark=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        if model.endswith("-global") and resolution == "4k":
            raise ValueError("doubao-seedance-2-5-260628-global 不支持 4k 分辨率，最高 1080p")

        content = [{"type": "text", "text": prompt}]
        content.extend(self._image_content("first_frame", first_frame_image, first_frame_url, "first_frame"))
        content.extend(self._image_content("last_frame", last_frame_image, last_frame_url, "last_frame"))
        content.extend(self._image_content("reference_image", reference_image, reference_image_url, "reference_image"))
        if reference_video_url and reference_video_url.strip():
            content.append({"type": "video_url", "video_url": {"url": reference_video_url.strip()}, "role": "reference_video"})
        if reference_audio_url and reference_audio_url.strip():
            content.append({"type": "audio_url", "audio_url": {"url": reference_audio_url.strip()}, "role": "reference_audio"})

        parameters = {
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "generate_audio": generate_audio,
            "camera_fixed": camera_fixed,
            "seed": seed,
            "watermark": watermark,
        }

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Doubao Seedance 2.5 task: model={model}")
        submit_res = mv_client.submit_task(model, {"content": content}, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Doubao Seedance 2.5 task submitted: {task_id}")
        video_url = self._poll_task(mv_client, task_id)
        return (video_url, task_id)

    @staticmethod
    def _image_content(role, image, url, label):
        has_url = bool(url and url.strip())
        has_image = image is not None
        if has_url and has_image:
            raise ValueError(f"请提供 {label}_image 或 {label}_url，不能同时提供")
        if has_url:
            return [{"type": "image_url", "image_url": {"url": url.strip()}, "role": role}]
        if has_image:
            return [{"type": "image_url", "image_url": {"url": _to_data_url(image)}, "role": role}]
        return []

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
    "DoubaoSeedance25": DoubaoSeedance25Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedance25": "Modelverse Doubao Seedance 2.5",
}

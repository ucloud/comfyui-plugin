"""
Kling Video O1 - 文生视频 / 图生视频 / 视频编辑
Model: kling-video-o1
"""
import time
from .modelverse_api.client import ModelverseClient
from .modelverse_api.utils import image_to_base64
from comfy.comfy_types.node_typing import IO


MODEL = "kling-video-o1"
MODES = ["pro"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
REFER_TYPES = ["feature", "base"]
YES_NO = ["yes", "no"]


def _to_image_value(image):
    data_url = image_to_base64(image)
    return data_url.split(",", 1)[1] if "," in data_url else data_url


class KlingVideoO1Node:
    """
    Kling Video O1。
    文本 + 可选首帧/尾帧图片（image_list）；可选参考视频（video_list）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "生成提示词"}),
            },
            "optional": {
                "first_frame_image": (IO.IMAGE, {"tooltip": "首帧图片（与 first_frame_url 二选一）"}),
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "首帧图片 URL"}),
                "last_frame_image": (IO.IMAGE, {"tooltip": "尾帧图片（与 last_frame_url 二选一）"}),
                "last_frame_url": (IO.STRING, {"default": "", "tooltip": "尾帧图片 URL"}),
                "video_url": (IO.STRING, {"default": "", "tooltip": "参考视频 URL（视频编辑/特征参考）"}),
                "refer_type": (REFER_TYPES, {"default": "feature", "tooltip": "参考视频用途: feature 特征参考 / base 视频编辑"}),
                "keep_original_sound": (YES_NO, {"default": "yes", "tooltip": "是否保留参考视频原声"}),
                "mode": (MODES, {"default": "pro", "tooltip": "O1 目前仅支持 pro"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": "输出宽高比"}),
                "duration": (IO.INT, {"default": 5, "min": 3, "max": 10, "step": 1,
                                      "tooltip": "文生/首帧图生仅支持 5 或 10 秒"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Kling"

    def generate(self, client, prompt, first_frame_image=None, first_frame_url="",
                 last_frame_image=None, last_frame_url="", video_url="",
                 refer_type="feature", keep_original_sound="yes", mode="pro",
                 aspect_ratio="16:9", duration=5):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        image_list = []
        image_list.extend(self._image_items(first_frame_image, first_frame_url, "first_frame", "first_frame"))
        image_list.extend(self._image_items(last_frame_image, last_frame_url, "end_frame", "last_frame"))

        parameters = {
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }
        if image_list:
            parameters["image_list"] = image_list

        if video_url and video_url.strip():
            parameters["video_list"] = [{
                "video_url": video_url.strip(),
                "refer_type": refer_type,
                "keep_original_sound": keep_original_sound,
            }]

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Kling Video O1 task: images={len(image_list)}, video={bool(video_url)}")
        submit_res = mv_client.submit_task(MODEL, {"prompt": prompt}, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Kling Video O1 task submitted: {task_id}")
        video_url_result = self._poll_task(mv_client, task_id)
        return (video_url_result, task_id)

    @staticmethod
    def _image_items(image, url, role, label):
        has_url = bool(url and url.strip())
        has_image = image is not None
        if has_url and has_image:
            raise ValueError(f"请提供 {label}_image 或 {label}_url，不能同时提供")
        if has_url:
            return [{"image_url": url.strip(), "type": role}]
        if has_image:
            return [{"image_url": _to_image_value(image), "type": role}]
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
    "Kling_Video_O1": KlingVideoO1Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kling_Video_O1": "Modelverse Kling Video O1",
}

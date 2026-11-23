"""
MiniMax H3 / H3-Max / H3-Lite - 多模态视频生成
Models: MiniMax-H3, MiniMax-H3-Max, minimax-h3-lite
NOTE: 仅接受公网可访问 URL，不接受 base64 / 本地文件。
"""
import time
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO


MODELS = ["MiniMax-H3", "MiniMax-H3-Max", "minimax-h3-lite"]
RESOLUTIONS = ["480P", "768P", "1080P", "2K", "4K"]
RATIOS = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]


class MiniMaxH3Node:
    """
    MiniMax H3 系列。所有素材必须是公网 URL；若接入本地 IMAGE 会直接报错提示改用 URL。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": "MiniMax-H3", "tooltip": "大小写敏感；H3-Max 不支持参考视频/音频"}),
                "prompt": (IO.STRING, {"multiline": True, "default": "", "tooltip": "文本提示词"}),
            },
            "optional": {
                "first_frame_url": (IO.STRING, {"default": "", "tooltip": "首帧图片公网 URL（不支持 base64）"}),
                "last_frame_url": (IO.STRING, {"default": "", "tooltip": "尾帧图片公网 URL（不支持 base64）"}),
                "reference_image_url": (IO.STRING, {"default": "", "tooltip": "参考图公网 URL（不支持 base64）"}),
                "reference_video_url": (IO.STRING, {"default": "", "tooltip": "参考视频公网 URL（H3-Max 不支持）"}),
                "reference_audio_url": (IO.STRING, {"default": "", "tooltip": "参考音频公网 URL（H3-Max 不支持）"}),
                "first_frame_image": (IO.IMAGE, {"tooltip": "不接受本地图片；请改用 first_frame_url"}),
                "last_frame_image": (IO.IMAGE, {"tooltip": "不接受本地图片；请改用 last_frame_url"}),
                "reference_image": (IO.IMAGE, {"tooltip": "不接受本地图片；请改用 reference_image_url"}),
                "resolution": (RESOLUTIONS, {"default": "768P", "tooltip": "H3: 768P/2K; H3-Max: 480P/768P; lite: 480P-4K"}),
                "duration": (IO.INT, {"default": 6, "min": 4, "max": 30, "step": 1,
                                      "tooltip": "时长(秒)，H3/H3-Max 为 4-15，lite 为 4-30"}),
                "ratio": (RATIOS, {"default": "16:9", "tooltip": "输出宽高比；纯文生时不能为 adaptive"}),
                "aigc_watermark": (IO.BOOLEAN, {"default": False, "tooltip": "是否添加 AIGC 水印（H3-Max 不支持）"}),
                "mute_audio": (IO.BOOLEAN, {"default": False, "tooltip": "仅 minimax-h3-lite 支持"}),
                "use_context_ir": (IO.BOOLEAN, {"default": False, "tooltip": "仅 minimax-h3-lite 支持"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("url", "task_id")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/MiniMax"

    def generate(self, client, model, prompt, first_frame_url="", last_frame_url="",
                 reference_image_url="", reference_video_url="", reference_audio_url="",
                 first_frame_image=None, last_frame_image=None, reference_image=None,
                 resolution="768P", duration=6, ratio="16:9",
                 aigc_watermark=False, mute_audio=False, use_context_ir=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")

        for name, img in (("first_frame", first_frame_image),
                          ("last_frame", last_frame_image),
                          ("reference_image", reference_image)):
            if img is not None:
                raise ValueError(
                    f"MiniMax 仅接受公网 URL，不支持本地 IMAGE/base64 输入。"
                    f"请勿连接 {name}_image，改用 {name}_url 提供公网可访问链接。"
                )

        content = [{"type": "text", "text": prompt}]
        for role, value in (("first_frame", first_frame_url),
                            ("last_frame", last_frame_url),
                            ("reference_image", reference_image_url)):
            if value and value.strip():
                content.append({"type": "image_url", "image_url": {"url": self._check_url(value, role)}, "role": role})
        if reference_video_url and reference_video_url.strip():
            content.append({"type": "video_url",
                            "video_url": {"url": self._check_url(reference_video_url, "reference_video")},
                            "role": "reference_video"})
        if reference_audio_url and reference_audio_url.strip():
            content.append({"type": "audio_url",
                            "audio_url": {"url": self._check_url(reference_audio_url, "reference_audio")},
                            "role": "reference_audio"})

        if model == "MiniMax-H3-Max":
            if reference_video_url or reference_audio_url:
                raise ValueError("MiniMax-H3-Max 不支持参考视频/音频，请移除对应输入")

        has_media = len(content) > 1
        if not has_media and ratio == "adaptive":
            raise ValueError("纯文生视频时 ratio 不能为 adaptive，请指定具体宽高比")

        parameters = {
            "resolution": resolution,
            "duration": duration,
            "ratio": ratio,
        }
        if model != "MiniMax-H3-Max":
            parameters["aigc_watermark"] = aigc_watermark
        if model == "minimax-h3-lite":
            parameters["mute_audio"] = mute_audio
            parameters["use_context_ir"] = use_context_ir

        mv_client = ModelverseClient(api_key)
        print(f"Submitting MiniMax H3 task: model={model}")
        submit_res = mv_client.submit_task(model, {"content": content}, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"MiniMax H3 task submitted: {task_id}")
        video_url = self._poll_task(mv_client, task_id)
        return (video_url, task_id)

    @staticmethod
    def _check_url(value, label):
        value = value.strip()
        if value.startswith("data:") or value.startswith("base64,"):
            raise ValueError(f"{label} 只接受公网 URL，不支持 base64/data URL")
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"{label} 必须是公网可访问的 http(s) URL")
        return value

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
    "MiniMax_H3": MiniMaxH3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMax_H3": "Modelverse MiniMax H3",
}

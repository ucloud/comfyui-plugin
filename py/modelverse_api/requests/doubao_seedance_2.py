"""
Doubao Seedance 2.0 task submit request builder.
API: https://www.compshare.cn/docs/modelverse/models/video_api/doubao-seedance-2-0
"""
from typing import Any, Dict, List, Optional

from pydantic import Field
from torch import Tensor

from ..utils import BaseRequest, image_to_base64


MODEL = "doubao-seedance-2-0-260128"
RESOLUTIONS = ["480p", "720p", "1080p"]
RATIOS = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]


def _resolve_image(image: Optional[Tensor], url: str, label: str) -> Optional[str]:
    has_url = url and url.strip()
    has_image = image is not None
    if has_url and has_image:
        raise ValueError(f"{label}: provide either image or url, not both")
    if has_url:
        return url.strip()
    if has_image:
        encoded = image_to_base64(image)
        if not encoded:
            raise ValueError(f"{label}: failed to convert image to base64")
        return encoded
    return None


def _normalize_prompt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        return ""
    return str(value).strip()


def _image_content(url: str, role: Optional[str] = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": url},
    }
    if role:
        item["role"] = role
    return item


class DoubaoSeedance2(BaseRequest):
    """
    Doubao Seedance 2.0 video generation task request.
    Supports text, first/last frame images, reference image/video/audio.
    """

    API_PATH = "/v1/tasks/submit"

    prompt: str = Field(default="", description="Text prompt for video generation")
    first_frame: Optional[Tensor] = Field(default=None, description="First frame image tensor")
    first_frame_url: str = Field(default="", description="First frame image URL or base64")
    last_frame: Optional[Tensor] = Field(default=None, description="Last frame image tensor")
    last_frame_url: str = Field(default="", description="Last frame image URL or base64")
    reference_image: Optional[Tensor] = Field(default=None, description="Reference image tensor")
    reference_image_url: str = Field(default="", description="Reference image URL or base64")
    reference_video_url: str = Field(default="", description="Reference video URL")
    reference_audio_url: str = Field(default="", description="Reference audio URL")
    duration: int = Field(default=5, ge=4, le=15, description="Video duration in seconds")
    resolution: str = Field(default="720p", description="Output resolution")
    ratio: str = Field(default="adaptive", description="Aspect ratio")
    seed: int = Field(default=0, ge=0, le=2147483647, description="Random seed")
    generate_audio: bool = Field(default=False, description="Whether to generate synchronized audio")
    camera_fixed: bool = Field(default=False, description="Whether to fix camera position")
    watermark: bool = Field(default=False, description="Whether to add watermark")
    draft: bool = Field(default=False, description="Draft mode (480p only)")

    def __init__(
        self,
        prompt: str = "",
        first_frame: Optional[Tensor] = None,
        first_frame_url: str = "",
        last_frame: Optional[Tensor] = None,
        last_frame_url: str = "",
        reference_image: Optional[Tensor] = None,
        reference_image_url: str = "",
        reference_video_url: str = "",
        reference_audio_url: str = "",
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "adaptive",
        seed: int = 0,
        generate_audio: bool = False,
        camera_fixed: bool = False,
        watermark: bool = False,
        draft: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt = _normalize_prompt(prompt)
        self.first_frame = first_frame
        self.first_frame_url = first_frame_url
        self.last_frame = last_frame
        self.last_frame_url = last_frame_url
        self.reference_image = reference_image
        self.reference_image_url = reference_image_url
        self.reference_video_url = reference_video_url
        self.reference_audio_url = reference_audio_url
        self.duration = duration
        self.resolution = resolution
        self.ratio = ratio
        self.seed = seed
        self.generate_audio = generate_audio
        self.camera_fixed = camera_fixed
        self.watermark = watermark
        self.draft = draft

    def build_content(self) -> List[Dict[str, Any]]:
        if self.draft and self.resolution != "480p":
            raise ValueError("Draft mode only supports 480p resolution")

        first_url = _resolve_image(self.first_frame, self.first_frame_url, "First frame")
        last_url = _resolve_image(self.last_frame, self.last_frame_url, "Last frame")
        ref_img_url = _resolve_image(self.reference_image, self.reference_image_url, "Reference image")

        content: List[Dict[str, Any]] = []
        if self.prompt and self.prompt.strip():
            content.append({"type": "text", "text": self.prompt.strip()})

        frame_images = [(first_url, "first_frame"), (last_url, "last_frame")]
        frame_images = [(url, role) for url, role in frame_images if url]
        if ref_img_url:
            frame_images.append((ref_img_url, "reference_image"))

        for url, role in frame_images:
            content.append(_image_content(url, role))

        if self.reference_video_url and self.reference_video_url.strip():
            content.append({
                "type": "video_url",
                "video_url": {"url": self.reference_video_url.strip()},
                "role": "reference_video",
            })

        if self.reference_audio_url and self.reference_audio_url.strip():
            content.append({
                "type": "audio_url",
                "audio_url": {"url": self.reference_audio_url.strip()},
                "role": "reference_audio",
            })

        if not content:
            raise ValueError(
                "At least a text prompt or one input asset (image/video/audio) is required. "
                f"Current prompt is empty (received: {self.prompt!r})"
            )
        return content

    def build_parameters(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "resolution": self.resolution,
            "ratio": self.ratio,
            "seed": self.seed,
            "generate_audio": self.generate_audio,
            "camera_fixed": self.camera_fixed,
            "watermark": self.watermark,
            "draft": self.draft,
        }

    def build_payload(self) -> Dict[str, Any]:
        return {
            "model": MODEL,
            "input": {"content": self.build_content()},
            "parameters": self.build_parameters(),
        }

    def field_required(self):
        return []

    def field_order(self):
        return [
            "prompt",
            "first_frame",
            "last_frame",
            "reference_image",
            "reference_video_url",
            "reference_audio_url",
            "duration",
            "resolution",
            "ratio",
            "seed",
            "generate_audio",
            "camera_fixed",
            "watermark",
            "draft",
        ]

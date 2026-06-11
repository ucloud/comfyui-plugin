"""
Kling V3 Omni task submit request builder.
Uses image_list / video_list / element_list parameters.
"""
from typing import Any, Dict, List, Optional

from pydantic import Field
from torch import Tensor

from ..utils import BaseRequest
from .kling_common import (
    ASPECT_RATIOS,
    MODEL_KLING_V3_OMNI,
    MODES,
    REFER_TYPES,
    SOUNDS,
    YES_NO,
    normalize_prompt,
    resolve_image,
)


class KlingV3Omni(BaseRequest):
    """
    Kling V3 Omni multimodal video generation task request.
    Supports reference images, videos, and element library IDs.
    """

    API_PATH = "/v1/tasks/submit"

    prompt: str = Field(default="", description="Text prompt for video generation")
    negative_prompt: str = Field(default="", description="Negative prompt")
    first_frame: Optional[Tensor] = Field(default=None, description="First frame image tensor")
    first_frame_url: str = Field(default="", description="First frame image URL or base64")
    last_frame: Optional[Tensor] = Field(default=None, description="Last frame image tensor")
    last_frame_url: str = Field(default="", description="Last frame image URL or base64")
    reference_video_url: str = Field(default="", description="Reference video URL")
    refer_type: str = Field(default="feature", description="Video reference type: feature or base")
    keep_original_sound: str = Field(default="no", description="Keep original video sound: yes or no")
    element_id: int = Field(default=0, description="Element library ID (0 to skip)")
    aspect_ratio: str = Field(default="16:9", description="Aspect ratio")
    duration: int = Field(default=5, ge=3, le=15, description="Video duration in seconds")
    mode: str = Field(default="std", description="Generation mode: std or pro")
    sound: str = Field(default="off", description="Generate synchronized audio: on or off")

    def __init__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        first_frame: Optional[Tensor] = None,
        first_frame_url: str = "",
        last_frame: Optional[Tensor] = None,
        last_frame_url: str = "",
        reference_video_url: str = "",
        refer_type: str = "feature",
        keep_original_sound: str = "no",
        element_id: int = 0,
        aspect_ratio: str = "16:9",
        duration: int = 5,
        mode: str = "std",
        sound: str = "off",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt = normalize_prompt(prompt)
        self.negative_prompt = normalize_prompt(negative_prompt)
        self.first_frame = first_frame
        self.first_frame_url = first_frame_url
        self.last_frame = last_frame
        self.last_frame_url = last_frame_url
        self.reference_video_url = reference_video_url
        self.refer_type = refer_type
        self.keep_original_sound = keep_original_sound
        self.element_id = element_id
        self.aspect_ratio = aspect_ratio
        self.duration = duration
        self.mode = mode
        self.sound = sound

    def build_input(self) -> Dict[str, Any]:
        task_input: Dict[str, Any] = {}
        if self.prompt:
            task_input["prompt"] = self.prompt
        if self.negative_prompt:
            task_input["negative_prompt"] = self.negative_prompt
        if not task_input.get("prompt"):
            raise ValueError(
                "Prompt is required for Kling V3 Omni. "
                f"Current prompt is empty (received: {self.prompt!r})"
            )
        return task_input

    def build_image_list(self) -> List[Dict[str, str]]:
        image_list: List[Dict[str, str]] = []
        first_url = resolve_image(self.first_frame, self.first_frame_url, "First frame")
        last_url = resolve_image(self.last_frame, self.last_frame_url, "Last frame")
        if first_url:
            image_list.append({"image_url": first_url, "type": "first_frame"})
        if last_url:
            image_list.append({"image_url": last_url, "type": "end_frame"})
        return image_list

    def build_parameters(self) -> Dict[str, Any]:
        has_video = bool(self.reference_video_url and self.reference_video_url.strip())
        if has_video and self.sound == "on":
            raise ValueError("sound must be 'off' when reference_video_url is provided")

        params: Dict[str, Any] = {
            "duration": self.duration,
            "aspect_ratio": self.aspect_ratio,
            "mode": self.mode,
            "sound": self.sound,
        }

        image_list = self.build_image_list()
        if image_list:
            params["image_list"] = image_list

        if has_video:
            params["video_list"] = [{
                "video_url": self.reference_video_url.strip(),
                "refer_type": self.refer_type,
                "keep_original_sound": self.keep_original_sound,
            }]

        if self.element_id:
            params["element_list"] = [{"element_id": self.element_id}]

        return params

    def build_payload(self) -> Dict[str, Any]:
        return {
            "model": MODEL_KLING_V3_OMNI,
            "input": self.build_input(),
            "parameters": self.build_parameters(),
        }

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return [
            "prompt",
            "negative_prompt",
            "first_frame",
            "last_frame",
            "reference_video_url",
            "refer_type",
            "keep_original_sound",
            "element_id",
            "aspect_ratio",
            "duration",
            "mode",
            "sound",
        ]

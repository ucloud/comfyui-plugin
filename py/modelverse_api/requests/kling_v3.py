"""
Kling V3 task submit request builder.
Routes to t2v / i2v / motion_control via input fields or parameters.kling_v3_type.
"""
from typing import Any, Dict, Optional

from pydantic import Field
from torch import Tensor

from ..utils import BaseRequest
from .kling_common import (
    ASPECT_RATIOS,
    CHARACTER_ORIENTATIONS,
    KLING_V3_TYPES,
    MODEL_KLING_V3,
    MODES,
    SOUNDS,
    YES_NO,
    normalize_prompt,
    resolve_image,
)


class KlingV3(BaseRequest):
    """
    Kling V3 unified video generation task request.
    Auto-routes to text-to-video, image-to-video, or motion control.
    """

    API_PATH = "/v1/tasks/submit"

    prompt: str = Field(default="", description="Text prompt for video generation")
    negative_prompt: str = Field(default="", description="Negative prompt")
    kling_v3_type: str = Field(
        default="auto",
        description="Task variant: auto, t2v, i2v, motion_control",
    )
    first_frame: Optional[Tensor] = Field(default=None, description="First frame image tensor")
    first_frame_url: str = Field(default="", description="First frame image URL or base64")
    last_frame: Optional[Tensor] = Field(default=None, description="Last frame image tensor")
    last_frame_url: str = Field(default="", description="Last frame image URL or base64")
    reference_video_url: str = Field(default="", description="Reference video URL for motion control")
    aspect_ratio: str = Field(default="16:9", description="Aspect ratio")
    duration: int = Field(default=5, ge=3, le=15, description="Video duration in seconds")
    mode: str = Field(default="std", description="Generation mode: std or pro")
    sound: str = Field(default="off", description="Generate synchronized audio: on or off")
    shot_type: str = Field(default="", description="Shot type, e.g. multi for multi-shot")
    character_orientation: str = Field(
        default="image",
        description="Motion control orientation: image or video",
    )
    keep_original_sound: str = Field(
        default="no",
        description="Keep original sound from reference video: yes or no",
    )

    def __init__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        kling_v3_type: str = "auto",
        first_frame: Optional[Tensor] = None,
        first_frame_url: str = "",
        last_frame: Optional[Tensor] = None,
        last_frame_url: str = "",
        reference_video_url: str = "",
        aspect_ratio: str = "16:9",
        duration: int = 5,
        mode: str = "std",
        sound: str = "off",
        shot_type: str = "",
        character_orientation: str = "image",
        keep_original_sound: str = "no",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt = normalize_prompt(prompt)
        self.negative_prompt = normalize_prompt(negative_prompt)
        self.kling_v3_type = kling_v3_type
        self.first_frame = first_frame
        self.first_frame_url = first_frame_url
        self.last_frame = last_frame
        self.last_frame_url = last_frame_url
        self.reference_video_url = reference_video_url
        self.aspect_ratio = aspect_ratio
        self.duration = duration
        self.mode = mode
        self.sound = sound
        self.shot_type = shot_type
        self.character_orientation = character_orientation
        self.keep_original_sound = keep_original_sound

    def _is_motion_control(self) -> bool:
        if self.kling_v3_type == "motion_control":
            return True
        if self.kling_v3_type == "auto" and self.reference_video_url and self.reference_video_url.strip():
            return True
        return False

    def build_input(self) -> Dict[str, Any]:
        task_input: Dict[str, Any] = {}
        if self.prompt:
            task_input["prompt"] = self.prompt
        if self.negative_prompt:
            task_input["negative_prompt"] = self.negative_prompt

        first_url = resolve_image(self.first_frame, self.first_frame_url, "First frame")
        last_url = resolve_image(self.last_frame, self.last_frame_url, "Last frame")

        if self._is_motion_control():
            if not first_url:
                raise ValueError("Motion control requires a first frame image or URL")
            if not self.reference_video_url or not self.reference_video_url.strip():
                raise ValueError("Motion control requires reference_video_url")
            task_input["first_frame_url"] = first_url
            task_input["video_url"] = self.reference_video_url.strip()
            return task_input

        if first_url:
            task_input["first_frame_url"] = first_url
        if last_url:
            task_input["last_frame_url"] = last_url

        if not task_input.get("prompt") and not first_url:
            raise ValueError(
                "At least a prompt or first frame image is required. "
                f"Current prompt is empty (received: {self.prompt!r})"
            )
        return task_input

    def build_parameters(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "duration": self.duration,
            "aspect_ratio": self.aspect_ratio,
            "mode": self.mode,
            "sound": self.sound,
        }
        if self.kling_v3_type and self.kling_v3_type != "auto":
            params["kling_v3_type"] = self.kling_v3_type
        if self.shot_type:
            params["shot_type"] = self.shot_type
        if self._is_motion_control():
            params["character_orientation"] = self.character_orientation
            params["keep_original_sound"] = self.keep_original_sound
        return params

    def build_payload(self) -> Dict[str, Any]:
        return {
            "model": MODEL_KLING_V3,
            "input": self.build_input(),
            "parameters": self.build_parameters(),
        }

    def field_required(self):
        return []

    def field_order(self):
        return [
            "prompt",
            "negative_prompt",
            "kling_v3_type",
            "first_frame",
            "last_frame",
            "reference_video_url",
            "aspect_ratio",
            "duration",
            "mode",
            "sound",
            "shot_type",
            "character_orientation",
            "keep_original_sound",
        ]

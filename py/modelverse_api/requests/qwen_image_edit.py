from typing import Optional
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor


class QwenImageEdit(BaseRequest):
    """
    Qwen/Qwen-Image-Edit request model for /v1/images/generations (OpenAI-compatible).
    Supports image edit with a prompt and optional controls.
    """

    API_PATH = "/v1/images/generations"

    prompt: str = Field(..., description="Prompt text for the edit")
    image: Tensor = Field(..., description="Input image tensor for edit")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")

    num_images: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images to generate (n)")
    strength: Optional[float] = Field(default=0.8, ge=0.0, le=1.0, description="Edit strength 0~1")
    seed: Optional[int] = Field(default=-1, description="Random seed (-1 for random)")
    steps: Optional[int] = Field(default=20, ge=1, le=50, description="Inference steps 1~50")
    guidance_scale: Optional[float] = Field(default=2.5, ge=1.0, le=10.0, description="Guidance scale 1~10")
    response_format: Optional[str] = Field(default="url", description='"url" or "b64_json"')

    def __init__(
        self,
        prompt: str,
        image: Tensor,
        negative_prompt: Optional[str] = "",
        num_images: Optional[int] = 1,
        strength: Optional[float] = 0.8,
        seed: Optional[int] = -1,
        steps: Optional[int] = 20,
        guidance_scale: Optional[float] = 2.5,
        response_format: Optional[str] = "url",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_images = num_images
        self.strength = strength
        self.seed = seed
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.response_format = response_format
        self.image = image_to_base64(image)

    def build_payload(self) -> dict:
        payload = {
            "model": "Qwen/Qwen-Image-Edit",
            "prompt": self.prompt,
            "image": self.image,
            "negative_prompt": self.negative_prompt,
            "n": self.num_images,
            "strength": self.strength,
            "seed": self.seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "response_format": self.response_format,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt", "image"]

    def field_order(self):
        return [
            "model",
            "prompt",
            "image",
            "negative_prompt",
            "n",
            "strength",
            "seed",
            "steps",
            "guidance_scale",
            "response_format",
        ]


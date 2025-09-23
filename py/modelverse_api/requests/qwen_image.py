from typing import Optional
from pydantic import Field
from ..utils import BaseRequest


class QwenImage(BaseRequest):
    """
    Qwen/Qwen-Image text-to-image via /v1/images/generations.
    """

    API_PATH = "/v1/images/generations"

    prompt: str = Field(..., description="Prompt text")
    aspect_ratio: Optional[str] = Field(default="1:1", description="Aspect ratio e.g. 16:9, 1:1")
    num_images: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images (n)")
    seed: Optional[int] = Field(default=-1, description="Random seed (-1 for random)")
    steps: Optional[int] = Field(default=20, ge=1, le=50, description="Inference steps")
    guidance_scale: Optional[float] = Field(default=2.5, ge=1.0, le=10.0, description="Guidance scale 1~10")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    response_format: Optional[str] = Field(default="url", description='"url" or "b64_json"')

    def __init__(
        self,
        prompt: str,
        aspect_ratio: Optional[str] = "1:1",
        num_images: Optional[int] = 1,
        seed: Optional[int] = -1,
        steps: Optional[int] = 20,
        guidance_scale: Optional[float] = 2.5,
        negative_prompt: Optional[str] = "",
        response_format: Optional[str] = "url",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.aspect_ratio = aspect_ratio
        self.num_images = num_images
        self.seed = seed
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.response_format = response_format

    def build_payload(self) -> dict:
        payload = {
            "model": "Qwen/Qwen-Image",
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio,
            "n": self.num_images,
            "seed": self.seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "response_format": self.response_format,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return [
            "model",
            "prompt",
            "aspect_ratio",
            "n",
            "seed",
            "steps",
            "guidance_scale",
            "negative_prompt",
            "response_format",
        ]


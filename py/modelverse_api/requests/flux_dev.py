from typing import Optional
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor

class FluxDev(BaseRequest):
    """
    Flux-dev text to image model, 12 billion parameter rectified flow transformer
    """
    guidance_scale: Optional[float] = Field(default=2.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you.")
    num_images: Optional[int] = Field(default=1, description="The number of images to generate.")
    num_inference_steps: Optional[int] = Field(default=28, description="The number of inference steps to perform.")
    prompt: str = Field(..., description="The prompt to generate an image from.")
    image: Optional[Tensor] = Field(default=None, description="The image for reference.")
    seed: Optional[int] = Field(default=-1, description="\n            The same seed and the same prompt given to the same version of the model\n            will output the same image every time.\n        ")
    width: Optional[int] = Field(default=1024, description="The width of the generated image." , ge=256, le=1536)
    height: Optional[int] = Field(default=1024, description="The height of the generated image." , ge=256, le=1536)
    strength: Optional[float] = Field(default=0.8, description="Strength indicates extent to transform the reference image", ge=0, le=1)

    def __init__(
            self,
            prompt: str,
            image: Optional[Tensor] = None,
            guidance_scale: Optional[float] = 2.5,
            num_images: Optional[int] = 1,
            num_inference_steps: Optional[int] = 28,
            seed: Optional[int] = -1,
            width: Optional[int] = 1024,
            height: Optional[int] = 1024,
            strength: Optional[float] = 0.8,
            **kwargs):
        super().__init__(**kwargs)
        
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_images = num_images
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.width = width
        self.height = height
        self.strength = strength
        self.image = image_to_base64(image) if isinstance(image, Tensor) else None

    def build_payload(self) -> dict:
        """Builds the request payload dictionary."""
        payload = {
            "model": "black-forest-labs/flux.1-dev",
            "prompt": self.prompt,
            "image": self.image,
            "guidance_scale": self.guidance_scale,
            "n": self.num_images,
            "steps": self.num_inference_steps,
            "seed": self.seed,
            "size": f"{self.width}x{self.height}",
            "strength": self.strength,
            "response_format": "url"
        }
        return self._remove_empty_fields(payload)

    # def get_api_path(self):
    #     """Gets the API path. Corresponds to api_path in the JSON."""
    #     return "/api/v3/images/generations"

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        """Corresponds to x-order-properties in the JSON request_schema."""
        return ["prompt", "image", "strength", "size", "num_inference_steps", "seed", "guidance_scale", "n", "response_format"]

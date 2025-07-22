from typing import Optional
from pydantic import Field
from ..utils import BaseRequest, image_to_base64
from torch import Tensor


class FluxKontextPro(BaseRequest):
    """
    Flux Kontext Pro for image editing.
    """
    prompt: str = Field(...,
                        description="The prompt to generate an image from.")
    image: Tensor = Field(...,
                          description="The image to generate an image from.")
    guidance_scale: Optional[float] = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you.")
    seed: Optional[int] = Field(
        default=-1, description="\n            The same seed and the same prompt given to the same version of the model\n            will output the same image every time.\n        ")

    def __init__(
            self,
            prompt: str,
            image: Tensor,
            guidance_scale: Optional[float] = 3.5,
            seed: Optional[int] = -1,
            **kwargs):
        super().__init__(**kwargs)

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.seed = seed

        self.image = image_to_base64(
            image) if isinstance(image, Tensor) else None

    def build_payload(self) -> dict:
        """Builds the request payload dictionary."""

        payload = {
            "model": "black-forest-labs/flux-kontext-pro",
            "prompt": self.prompt,
            "image": self.image,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }
        return self._remove_empty_fields(payload)

    # def get_api_path(self):
    #     """Gets the API path. Corresponds to api_path in the JSON."""
    #     return "/api/v3/images/generations"

    def field_required(self):
        return ["prompt", "image"]

    def field_order(self):
        """Corresponds to x-order-properties in the JSON request_schema."""
        return ["model", "prompt", "image", "guidance_scale", "seed", "response_format"]


class FluxKontextProMulti(BaseRequest):
    """
    Flux Kontext Pro for multiple image inputs.
    """
    prompt: str = Field(...,
                        description="The prompt to generate an image from.")
    images: list[Tensor] = Field(...,
                                 description="The image to generate an image from.")
    guidance_scale: Optional[float] = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you.")
    seed: Optional[int] = Field(
        default=-1, description="\n            The same seed and the same prompt given to the same version of the model\n            will output the same image every time.\n        ")

    def __init__(
            self,
            prompt: str,
            images: list[Tensor],
            guidance_scale: Optional[float] = 3.5,
            seed: Optional[int] = -1,
            **kwargs):
        super().__init__(**kwargs)

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.images = [image_to_base64(
            image) for image in images if isinstance(image, Tensor)]

    def build_payload(self) -> dict:
        """Builds the request payload dictionary."""

        payload = {
            "model": "black-forest-labs/flux-kontext-pro/multi",
            "prompt": self.prompt,
            "images": self.images,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }
        return self._remove_empty_fields(payload)

    # def get_api_path(self):
    #     """Gets the API path. Corresponds to api_path in the JSON."""
    #     return "/api/v3/images/generations"

    def field_required(self):
        return ["prompt", "images"]

    def field_order(self):
        """Corresponds to x-order-properties in the JSON request_schema."""
        return ["model", "prompt", "images", "guidance_scale", "seed", "response_format"]


class FluxKontextProT2I(BaseRequest):
    """
    Flux Kontext Pro text-to-image model
    """
    prompt: str = Field(...,
                        description="The prompt to generate an image from.")
    aspect_ratio: Optional[str] = Field(
        default="1:1", description="The aspect ratio of the output image, ranging from \"21:9\" to \"9:21\", default is \"1:1\"."),
    num_images: Optional[int] = Field(
        default=1, description="The number of images to generate.")
    guidance_scale: Optional[float] = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you.")
    seed: Optional[int] = Field(
        default=-1, description="\n            The same seed and the same prompt given to the same version of the model\n            will output the same image every time.\n        ")

    def __init__(
            self,
            prompt: str,
            aspect_ratio: Optional[str] = "1:1",
            num_images: Optional[int] = 1,
            guidance_scale: Optional[float] = 3.5,
            seed: Optional[int] = -1,
            **kwargs):
        super().__init__(**kwargs)

        self.prompt = prompt
        self.num_images = num_images
        self.guidance_scale = guidance_scale
        self.aspect_ratio = aspect_ratio
        self.seed = seed

    def build_payload(self) -> dict:
        """Builds the request payload dictionary."""

        payload = {
            "model": "black-forest-labs/flux-kontext-pro/text-to-image",
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio,
            "n": self.num_images,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }
        return self._remove_empty_fields(payload)

    # def get_api_path(self):
    #     """Gets the API path. Corresponds to api_path in the JSON."""
    #     return "/api/v3/images/generations"

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        """Corresponds to x-order-properties in the JSON request_schema."""
        return ["model", "prompt", "aspect_ratio", "n", "guidance_scale", "seed", "response_format"]

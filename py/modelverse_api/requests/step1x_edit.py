from typing import Optional, Union
from pydantic import BaseModel, Field
from ..utils import BaseRequest, image_to_base64, normalization_loras  # normalization_loras is not used in this specific class but kept for consistency with the example if other similar classes might need it.
from torch import Tensor

class Step1xEdit(BaseRequest):
    """
    Request model for the Step1X Edit API.

    Step1X-Edit transforms your photos with simple instructions into stunning, professional-quality edits—rivaling top proprietary tools.
    """
    prompt: str = Field(..., description="The prompt to generate an image from.")
    image: Tensor = Field(..., description="The image to generate an image from. Needs to match the dimensions of the mask.")
    negative_prompt: Optional[str] = Field(
        default="",
        description=
        "\n            The negative prompt to use. Use it to address details that you don't want\n            in the image. This could be colors, objects, scenery and even the small details\n            (e.g. moustache, blurry, low resolution).\n        "
    )
    seed: Optional[int] = Field(default=-1, description="\n            The same seed and the same prompt given to the same version of the model\n            will output the same image every time.\n        "
    )  # JSON has no default, using None as it's optional. Example used -1.
    guidance_scale: Optional[float] = Field(
        default=4.0,
        ge=0,
        le=20,
        description="\n            The CFG (Classifier Free Guidance) scale is a measure of how close you want\n            the model to stick to your prompt when looking for a related image to show you.\n        ")
    num_inference_steps: Optional[int] = Field(default=30, ge=1, le=50, description="The number of inference steps to perform.")

    def __init__(
            self,
            prompt: str,
            image: Tensor,
            negative_prompt: Optional[str] = "",
            seed: Optional[int] = -1,  # Matching Field default
            guidance_scale: Optional[float] = 4.0,
            num_inference_steps: Optional[int] = 30,
            **kwargs):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.image = image_to_base64(image)

    def build_payload(self) -> dict:
        """Builds the request payload dictionary."""
        payload = {
            "model": "stepfun-ai/step1x-edit",
            "prompt": self.prompt,
            "image": self.image,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }
        return self._remove_empty_fields(payload)

    # def get_api_path(self):
    #     """Gets the API path for the request. Corresponds to api_path in the interface configuration json"""
    #     return "/api/v3/images/generations"

    def field_required(self):
        """Corresponds to required fields in the interface configuration json"""
        return ["prompt", "image"]

    def field_order(self):        
        """Corresponds to x-fal-order-properties in the interface configuration json"""
        return ["model", "prompt", "image", "negative_prompt", "num_inference_steps", "guidance_scale", "seed", "response_format"]

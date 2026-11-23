"""
Tripo 3D - text / image / multiview to 3D model generation (tripo-h3.1, tripo-p2)

Docs:
  docs/openapi/chinese/model3d/tripo-h3.1.yaml
  docs/openapi/chinese/model3d/tripo-p2.yaml
"""
import os
import time
from urllib.parse import urlparse

import folder_paths
import requests
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient


MODELS = ["tripo-h3.1", "tripo-p2"]
TASK_TYPES = ["text-to-model", "image-to-model", "multiview-to-model"]
TEXTURE_QUALITIES = ["standard", "detailed", "extreme"]
GEOMETRY_QUALITIES = ["standard", "detailed", "ultra"]

PROMPT_MAX_LEN = 1024
NEGATIVE_PROMPT_MAX_LEN = 255

# tripo-p2: face_limit 48~50,000（三角面）/ 48~25,000（四边面）
P2_FACE_LIMIT_RANGE = (48, 50000)
P2_QUAD_FACE_LIMIT_RANGE = (48, 25000)
# tripo-h3.1: 标准几何最多 1,500,000，超清几何最多 2,000,000，四边面最多 150,000
H31_FACE_LIMIT_MAX = {"standard": 1500000, "detailed": 2000000, "ultra": 2000000}
H31_QUAD_FACE_LIMIT_MAX = 150000


class Tripo3DNode:
    """
    Modelverse Tripo 3D generation.
    Requires public HTTP(S) image URLs, as defined by the API (input.img_url / input.views).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {
                    "default": "tripo-h3.1",
                    "tooltip": "Model id: tripo-h3.1 or tripo-p2",
                }),
                "task_type": (TASK_TYPES, {
                    "default": "text-to-model",
                    "tooltip": "parameters.type: text-to-model / image-to-model / multiview-to-model",
                }),
            },
            "optional": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "text-to-model 必填，最多 1024 字符（input.prompt）",
                }),
                "negative_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "仅 text-to-model 生效，最多 255 字符（input.negative_prompt）",
                }),
                "img_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "image-to-model 必填：公网可访问的 HTTP/HTTPS 图片直链（input.img_url）",
                }),
                "image": (IO.IMAGE, {
                    "tooltip": "image-to-model 的 IMAGE 输入；接口仅接受公网 URL，无法直接上传时请改用 img_url",
                }),
                "front_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "multiview-to-model 必填：正面视图公网 URL（views.front）",
                }),
                "left_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "multiview-to-model 可选：左侧视图公网 URL（views.left）",
                }),
                "back_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "multiview-to-model 可选：背面视图公网 URL（views.back）",
                }),
                "right_url": (IO.STRING, {
                    "default": "",
                    "tooltip": "multiview-to-model 可选：右侧视图公网 URL（views.right）",
                }),
                "texture": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "parameters.texture：是否生成贴图；与 pbr 同为 false 时输出白模",
                }),
                "pbr": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "parameters.pbr：是否启用 PBR 材质贴图（开启时强制生成贴图）",
                }),
                "texture_quality": (TEXTURE_QUALITIES, {
                    "default": "standard",
                    "tooltip": "parameters.texture_quality：standard / detailed / extreme(8K)",
                }),
                "geometry_quality": (GEOMETRY_QUALITIES, {
                    "default": "standard",
                    "tooltip": "parameters.geometry_quality：仅 tripo-h3.1 支持（ultra 等同 detailed）",
                }),
                "face_limit": (IO.INT, {
                    "default": 0, "min": 0, "max": 2000000, "step": 1,
                    "tooltip": "parameters.face_limit：最大面数，0 表示不传（自适应）",
                }),
                "quad": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "parameters.quad：输出四边面网格（输出格式强制为 FBX）",
                }),
                "export_uv": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "parameters.export_uv：生成过程中是否展开 UV",
                }),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("model_path", "model_url")
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/3D"

    def generate(
        self,
        client,
        model="tripo-h3.1",
        task_type="text-to-model",
        prompt="",
        negative_prompt="",
        img_url="",
        image=None,
        front_url="",
        left_url="",
        back_url="",
        right_url="",
        texture=True,
        pbr=True,
        texture_quality="standard",
        geometry_quality="standard",
        face_limit=0,
        quad=False,
        export_uv=True,
    ):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set")
        if model not in MODELS:
            raise ValueError(f"Unsupported model: {model!r}; expected one of {MODELS}")
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {task_type!r}; expected one of {TASK_TYPES}")

        task_input = self._build_input(
            task_type=task_type,
            prompt=prompt,
            negative_prompt=negative_prompt,
            img_url=img_url,
            image=image,
            front_url=front_url,
            left_url=left_url,
            back_url=back_url,
            right_url=right_url,
        )
        parameters = self._build_parameters(
            model=model,
            task_type=task_type,
            texture=texture,
            pbr=pbr,
            texture_quality=texture_quality,
            geometry_quality=geometry_quality,
            face_limit=face_limit,
            quad=quad,
            export_uv=export_uv,
        )

        mv_client = ModelverseClient(api_key)
        print(f"Submitting Tripo 3D task: model={model}, type={task_type}")
        submit_res = mv_client.submit_task(model, task_input, parameters)
        task_id = submit_res.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"Failed to submit task: {submit_res}")

        print(f"Tripo 3D task submitted: {task_id}")
        model_url = self._poll_task(mv_client, task_id, max_retries=240)
        model_path = self._download_model(model_url, model=model, quad=quad)
        print(f"Tripo 3D model saved: {model_path}")
        return (model_path, model_url)

    @staticmethod
    def _check_public_url(url, label):
        value = (url or "").strip()
        if not value:
            raise ValueError(f"{label} is required")
        if not value.lower().startswith(("http://", "https://")):
            raise ValueError(
                f"{label} must be a public HTTP/HTTPS image URL (got: {value[:80]!r})"
            )
        return value

    def _build_input(
        self,
        task_type,
        prompt,
        negative_prompt,
        img_url,
        image,
        front_url,
        left_url,
        back_url,
        right_url,
    ):
        if task_type == "text-to-model":
            text = (prompt or "").strip()
            if not text:
                raise ValueError("text-to-model requires a non-empty prompt (input.prompt)")
            if len(text) > PROMPT_MAX_LEN:
                raise ValueError(
                    f"prompt is too long: {len(text)} characters (max {PROMPT_MAX_LEN})"
                )
            task_input = {"prompt": text}
            negative = (negative_prompt or "").strip()
            if negative:
                if len(negative) > NEGATIVE_PROMPT_MAX_LEN:
                    raise ValueError(
                        f"negative_prompt is too long: {len(negative)} characters "
                        f"(max {NEGATIVE_PROMPT_MAX_LEN})"
                    )
                task_input["negative_prompt"] = negative
            return task_input

        if task_type == "image-to-model":
            url = (img_url or "").strip()
            if image is not None and url:
                raise ValueError(
                    "image-to-model: provide either `image` or `img_url`, not both"
                )
            if image is not None and not url:
                raise ValueError(
                    "image-to-model only accepts a public HTTP/HTTPS image URL "
                    "(input.img_url); a local IMAGE tensor cannot be uploaded here. "
                    "Please host the image and pass it via `img_url`."
                )
            return {"img_url": self._check_public_url(url, "image-to-model img_url")}

        # multiview-to-model
        if image is not None:
            raise ValueError(
                "multiview-to-model only accepts public HTTP/HTTPS image URLs "
                "(input.views); IMAGE input is not supported in this mode. "
                "Please use front_url/left_url/back_url/right_url."
            )
        candidates = [
            ("front", front_url),
            ("left", left_url),
            ("back", back_url),
            ("right", right_url),
        ]
        views = []
        for key, url in candidates:
            value = (url or "").strip()
            if not value:
                continue
            views.append({key: self._check_public_url(value, f"multiview {key} view URL")})
        if not views:
            raise ValueError(
                "multiview-to-model requires at least 2 view URLs, and the front view is mandatory"
            )
        if not any("front" in view for view in views):
            raise ValueError("multiview-to-model requires the front view URL (front_url)")
        if len(views) < 2:
            raise ValueError(
                f"multiview-to-model requires at least 2 views, got {len(views)}"
            )
        return {"views": views}

    @staticmethod
    def _build_parameters(
        model,
        task_type,
        texture,
        pbr,
        texture_quality,
        geometry_quality,
        face_limit,
        quad,
        export_uv,
    ):
        if texture_quality not in TEXTURE_QUALITIES:
            raise ValueError(
                f"Unsupported texture_quality: {texture_quality!r}; "
                f"expected one of {TEXTURE_QUALITIES}"
            )

        parameters = {
            "type": task_type,
            "texture": bool(texture),
            "pbr": bool(pbr),
            "texture_quality": texture_quality,
            "quad": bool(quad),
            "export_uv": bool(export_uv),
        }

        if model == "tripo-h3.1":
            # geometry_quality 仅 tripo-h3.1 的参数定义中存在
            if geometry_quality not in GEOMETRY_QUALITIES:
                raise ValueError(
                    f"Unsupported geometry_quality: {geometry_quality!r}; "
                    f"expected one of {GEOMETRY_QUALITIES}"
                )
            parameters["geometry_quality"] = geometry_quality

        if face_limit and face_limit > 0:
            if model == "tripo-p2":
                low, high = P2_QUAD_FACE_LIMIT_RANGE if quad else P2_FACE_LIMIT_RANGE
                if not low <= face_limit <= high:
                    raise ValueError(
                        f"tripo-p2 face_limit must be within {low}-{high}"
                        f"{' (quad=true)' if quad else ''}, got {face_limit}"
                    )
            else:
                high = (
                    H31_QUAD_FACE_LIMIT_MAX
                    if quad
                    else H31_FACE_LIMIT_MAX[geometry_quality]
                )
                if face_limit > high:
                    raise ValueError(
                        f"tripo-h3.1 face_limit must be at most {high}"
                        f"{' (quad=true)' if quad else ''}, got {face_limit}"
                    )
            parameters["face_limit"] = int(face_limit)

        return parameters

    @staticmethod
    def _poll_task(mv_client, task_id, max_retries=240):
        for i in range(max_retries):
            status_res = mv_client.get_task_status(task_id)
            output = status_res.get("output", {})
            task_status = output.get("task_status")

            if task_status == "Success":
                urls = output.get("urls") or []
                if urls:
                    return urls[0]
                raise Exception("Task succeeded but no model URL returned")
            if task_status in ("Failure", "Expired"):
                error = output.get("error_message", f"task status: {task_status}")
                raise Exception(f"Task failed: {error}")
            if task_status in ("Pending", "Running"):
                print(f"Task {task_id}: {task_status} ({i + 1}/{max_retries})")
                time.sleep(5)
                continue
            raise Exception(f"Unknown status: {task_status}")

        raise Exception("Task timed out")

    @staticmethod
    def _download_model(model_url, model, quad):
        suffix = os.path.splitext(urlparse(model_url).path)[1].lower().lstrip(".")
        if suffix not in ("glb", "gltf", "fbx", "obj", "usdz", "zip"):
            suffix = "fbx" if quad else "glb"

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"tripo3d_{model.replace('.', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
        file_path = os.path.join(output_dir, f"{base_name}.{suffix}")
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(output_dir, f"{base_name}_{counter}.{suffix}")
            counter += 1

        response = requests.get(model_url, timeout=600)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        return os.path.abspath(file_path)


NODE_CLASS_MAPPINGS = {
    "Modelverse_Tripo3D": Tripo3DNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_Tripo3D": "Modelverse Tripo 3D",
}

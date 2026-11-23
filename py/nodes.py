import os
import json
import stat
import configparser
from secrets import compare_digest, token_urlsafe

import torch
import server
from aiohttp import web
from comfy.comfy_types.node_typing import IO

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    config_path = os.path.join(parent_dir, 'config.ini')
    config = configparser.ConfigParser()

    if not os.path.exists(config_path):
        config['API'] = {'MODELVERSE_API_KEY': ''}
        with open(config_path, 'w') as config_file:
            config.write(config_file)

    config.read(config_path)
except Exception as e:
    print(f"Error reading or creating config file: {e}")
    config = None


secrets_path = os.path.join(parent_dir, 'secrets.json')

# Per-process token required for state-changing requests. It is handed to the
# ComfyUI web UI through /modelverse-secrets/token, so a cross-site request or a
# plain browser navigation cannot read, overwrite or delete stored API keys.
_secrets_token = token_urlsafe(32)


def _secure_write(path, data):
    """Write JSON with 0600 permissions (keys must never be world-readable)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, 'w') as secrets_file:
        json.dump(data, secrets_file, indent=2)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def load_secrets():
    if not os.path.exists(secrets_path):
        _secure_write(secrets_path, {})
    with open(secrets_path) as secrets_file:
        return json.load(secrets_file)


def save_secrets(secrets):
    _secure_write(secrets_path, secrets)


def _mask_secret(value):
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * 15
    return f"{value[:4]}{'*' * 15}"


def _is_same_origin(request):
    """Reject cross-site browser requests (CSRF / DNS rebinding)."""
    origin = request.headers.get("Origin")
    if origin is not None:
        if origin == "null":
            return False
        host = request.headers.get("Host", "")
        if origin.split("://", 1)[-1].split("/", 1)[0] != host:
            return False
    fetch_site = request.headers.get("Sec-Fetch-Site")
    if fetch_site is not None and fetch_site != "same-origin":
        return False
    return True


def _has_valid_token(request):
    provided = request.headers.get("X-Modelverse-Secret-Token", "")
    return compare_digest(provided.encode("utf-8", "surrogateescape"), _secrets_token.encode())


def _forbidden():
    return web.json_response({"error": "Forbidden"}, status=403)


async def get_modelverse_secrets_token(request):
    if not _is_same_origin(request):
        return _forbidden()
    return web.json_response({"token": _secrets_token})


async def get_modelverse_secrets(request):
    if not _is_same_origin(request):
        return _forbidden()
    return web.json_response({key: _mask_secret(value) for key, value in load_secrets().items()})


async def set_modelverse_secret(request):
    if not _is_same_origin(request) or not _has_valid_token(request):
        return _forbidden()
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid or empty request body"}, status=400)

    key = str(data.get("key", "")).strip()
    value = data.get("value", "")
    old_key = str(data.get("old_key", "")).strip()
    if not key:
        return web.json_response({"error": "Key cannot be empty"}, status=400)
    if not isinstance(value, str):
        return web.json_response({"error": "Value must be a string"}, status=400)

    secrets = load_secrets()
    if old_key and old_key != key:
        if old_key not in secrets:
            return web.json_response({"error": "Key not found"}, status=404)
        # Values are never sent to the browser, so a rename without a new value
        # moves the stored key instead of overwriting it with an empty string.
        stored_value = secrets.pop(old_key)
        value = value or stored_value
    elif not value and key not in secrets:
        return web.json_response({"error": "Value cannot be empty"}, status=400)

    if value:
        secrets[key] = value
    save_secrets(secrets)
    return web.json_response({"ok": True})


async def delete_modelverse_secret(request):
    if not _is_same_origin(request) or not _has_valid_token(request):
        return _forbidden()
    key = request.match_info["key"]
    secrets = load_secrets()
    if key not in secrets:
        return web.json_response({"error": "Key not found"}, status=404)

    del secrets[key]
    save_secrets(secrets)
    return web.json_response({"ok": True})


if not getattr(server.PromptServer.instance, "_modelverse_secrets_registered", False):
    server.PromptServer.instance.routes.get("/modelverse-secrets/token")(get_modelverse_secrets_token)
    server.PromptServer.instance.routes.get("/modelverse-secrets")(get_modelverse_secrets)
    server.PromptServer.instance.routes.post("/modelverse-secrets")(set_modelverse_secret)
    server.PromptServer.instance.routes.delete("/modelverse-secrets/{key}")(delete_modelverse_secret)
    server.PromptServer.instance._modelverse_secrets_registered = True


class ModelverseAPIClient:
    """
    Ucloud Modelverse API Client Node

    This node creates a client for connecting to the Ucloud Modelverse API.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "ModelVerse API key. Leave empty to read MODELVERSE_API_KEY from config.ini and avoid storing the key in workflow files."
                })
            },
        }

    RETURN_TYPES = ("MODELVERSE_API_CLIENT",)
    RETURN_NAMES = ("client",)

    FUNCTION = "create_client"

    CATEGORY = "UCLOUD_MODELVERSE"

    def create_client(self, api_key):
        """
        Create a UCloud Modelverse API client

        Args:
            api_key: UCloud Modelverse API key

        Returns:
            ModelVerseAPI: UCloud Modelverse API client
        """
        modelverse_api_key = ""
        if api_key == "":
            try:
                modelverse_api_key = config['API']['MODELVERSE_API_KEY']
                if modelverse_api_key == '':
                    raise ValueError('API_KEY is empty')

            except KeyError:
                raise ValueError('Unable to find API_KEY in config.ini')

        else:
            modelverse_api_key = api_key

        return ({
            "api_key": modelverse_api_key
        },)


class ModelverseSecretClient:
    """
    UCloud Modelverse API Client from a locally managed secret.

    This node stores only the secret name in the workflow. The actual API key is
    read from secrets.json at execution time.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "secret": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Secret name from Modelverse Secrets Manager. The workflow stores this name, not the API key."
                })
            },
        }

    RETURN_TYPES = ("MODELVERSE_API_CLIENT",)
    RETURN_NAMES = ("client",)

    FUNCTION = "create_client"

    CATEGORY = "UCLOUD_MODELVERSE"

    @classmethod
    def IS_CHANGED(cls, secret):
        return load_secrets().get(secret, "")

    def create_client(self, secret):
        secret = secret.strip() if isinstance(secret, str) else secret
        if not secret:
            raise ValueError("Secret name is required")

        secrets = load_secrets()
        api_key = secrets.get(secret)
        if not api_key:
            raise ValueError(f"Secret '{secret}' not found in secrets.json")

        return ({
            "api_key": api_key
        },)


class ModelverseImagePacker:
    """
    Ucloud Modelverse Image Packer

    This node packs multiple images into a batched IMAGE tensor for multi-image editing.

    Args:
        images1: The first image to be packed together with.
        images2: The second image to be packed together with, et cetera.

    Returns:
        images: batched IMAGE tensor for Flux Kontext Pro/Max multi-image mode.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images1": (IO.IMAGE, {"tooltip": "The first image/list to be packed together. Add more if you need."})
            },
            "optional": {
                "images2": (IO.IMAGE, {"default": None, "tooltip": "The second image/list to be packed together."}),
                "images3": (IO.IMAGE, {"default": None, "tooltip": "The third image/list to be packed together."}),
                "images4": (IO.IMAGE, {"default": None, "tooltip": "The fourth image/list to be packed together."}),
                "images5": (IO.IMAGE, {"default": None, "tooltip": "The fifth image/list to be packed together."})
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)

    FUNCTION = "pack_images"

    CATEGORY = "UCLOUD_MODELVERSE"

    def pack_images(self,
                    images1,
                    images2=None,
                    images3=None,
                    images4=None,
                    images5=None
                    ):

        # 把单个IMAGE转成单元素list,便于之后统一extend
        def to_list(x): return x if isinstance(x, list) else [x]

        result = []
        for i in (images1, images2, images3, images4, images5):
            if i is not None:
                result.extend(to_list(i))
        if not result:
            raise ValueError("At least one image is required")
        return (torch.cat(result, dim=0),)


NODE_CLASS_MAPPINGS = {
    'UCloud ModelVerse Client': ModelverseAPIClient,
    'UCloud ModelVerse Secret Client': ModelverseSecretClient,
    'ModelVerse Image Packer': ModelverseImagePacker
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'UCloud ModelVerse Client': 'Modelverse Client',
    'UCloud ModelVerse Secret Client': 'Modelverse Secret Client',
    'ModelVerse Image Packer': 'Modelverse Image Packer'
}

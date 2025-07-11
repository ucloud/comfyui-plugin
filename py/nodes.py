import os
import requests
import configparser
import folder_paths
import time
import torchaudio
from .modelverse_api.client import ModelverseClient
# from .modelverse_api.utils import tensor2images

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
                "api_key": ("STRING", {"multiline": False, "default": ""})
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

class ModelverseImagePacker:
    """
    Ucloud Modelverse Image Packer

    This node packs multiple images into a list for multi-image editing.

    Args:
        images1: The first image/image_list to be packed together with.
        images2: The second image/image_list to be packed together with. et cetera.

    Returns:
        image_list: the pack of all input images.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images1": ("IMAGE,IMAGE_LIST", {"tooltip":"The first image/list to be packed together. Add more if you need."})
            },
            "optional": {
                "images2": ("IMAGE,IMAGE_LIST", {"default": None, "tooltip":"The second image/list to be packed together."}),
                "images3": ("IMAGE,IMAGE_LIST", {"default": None, "tooltip":"The third image/list to be packed together."}),
                "images4": ("IMAGE,IMAGE_LIST", {"default": None, "tooltip":"The fourth image/list to be packed together."}),
                "images5": ("IMAGE,IMAGE_LIST", {"default": None, "tooltip":"The fifth image/list to be packed together."})
            }
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("image_list",)

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
        to_list = lambda x: x if isinstance(x, list) else [x]

        result = []
        for i in (images1, images2, images3, images4, images5):
            if i is not None:
                result.extend(to_list(i))
        return (result,)


NODE_CLASS_MAPPINGS = {
    'UCloud ModelVerse Client': ModelverseAPIClient,
    'ModelVerse Image Packer': ModelverseImagePacker
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'UCloud ModelVerse Client': 'Modelverse Client',
    'ModelVerse Image Packer': 'Modelverse Image Packer'
}

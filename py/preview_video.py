import os
import re
import requests
import folder_paths


class ModelversePreviewVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "Modelverse"}),
                "save_output": ("BOOLEAN", {"default": True}),
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "UCLOUD_MODELVERSE"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)

    def run(self, video_url, filename_prefix, save_output):
        if not save_output:
            return {"ui": {"video_url": [video_url]}, "result": ('',)}

        output_dir = folder_paths.get_output_directory()
        (
            full_output_folder,
            filename,
            _,
            _,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter

        counter = max_counter + 1
        file = f"{filename}_{counter:05}.mp4"
        file_path = os.path.join(full_output_folder, file)

        if type(video_url) == list:
            video_url = video_url[0]

        response = requests.get(video_url, stream=True)
        open(file_path, "wb").write(response.content)

        return {"ui": {"video_url": [video_url]}, "result": (file_path,)}


NODE_CLASS_MAPPINGS = {
    "Modelverse_PreviewVideo": ModelversePreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_PreviewVideo": "Modelverse Preview Video",
}

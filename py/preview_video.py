import os
import re
import folder_paths
from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.apis.client import ApiEndpoint, HttpMethod, SynchronousOperation
from comfy_api_nodes.util import download_url_to_video_output


class ModelversePreviewVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": (IO.STRING, {"forceInput": True}),
                "filename_prefix": (IO.STRING, {"default": "Modelverse"}),
                "save_output": (IO.BOOLEAN, {"default": True}),
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "UCLOUD_MODELVERSE"

    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)

    async def run(self, video_url, filename_prefix, save_output):
        if type(video_url) == list:
            video_url = video_url[0]

        # Download video and get VIDEO output
        video = await download_url_to_video_output(video_url)

        # Save to file if requested
        if save_output:
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

            # Get video data from the video object and save
            if hasattr(video, 'video_io'):
                video_data = video.video_io.getvalue()
            elif hasattr(video, 'get_bytes'):
                video_data = video.get_bytes()
            else:
                # Try to read from the BytesIO
                video.video_io.seek(0)
                video_data = video.video_io.read()
                video.video_io.seek(0)

            with open(file_path, "wb") as f:
                f.write(video_data)

        return {"ui": {"video_url": [video_url]}, "result": (video,)}


NODE_CLASS_MAPPINGS = {
    "Modelverse_PreviewVideo": ModelversePreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_PreviewVideo": "Modelverse Preview Video",
}

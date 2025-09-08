import os
import shutil
import folder_paths
from datetime import datetime

class SaveVideo:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "UCLOUD_MODELVERSE"

    def save_video(self, video_path, filename_prefix="ComfyUI"):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.mp4"
        
        # Save to ComfyUI output directory
        output_path = os.path.join(self.output_dir, filename)
        shutil.copy2(video_path, output_path)
        
        # Clean up temporary file
        try:
            os.unlink(video_path)
        except:
            pass  # Don't fail if we can't delete temp file
            
        return {"ui": {"videos": [{"filename": filename, "subfolder": "", "type": "output"}]}}

NODE_CLASS_MAPPINGS = {
    "SaveVideo": SaveVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVideo": "Save Video"
}

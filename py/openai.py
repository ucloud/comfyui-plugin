import io
import os
import json
import openai
from openai import OpenAI
import base64
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO
from server import PromptServer
import folder_paths

# Hardcoded models list
DEFAULT_MODELS = [
    'ByteDance/doubao-seed-1.6',
    'Qwen/QwQ-32B',
    'Qwen/Qwen-Image',
    'Qwen/Qwen-Image-Edit',
    'Qwen/Qwen3-32B',
    'Qwen/Qwen3-Coder',
    'doubao-seedance-2-0-260128',
    'kling-v3',
    'kling-v3-omni',
    'happyhorse-1.0-t2v',
    'happyhorse-1.0-i2v',
    'happyhorse-1.0-r2v',
    'veo-3.1-generate-001',
    'veo-3.1-fast-generate-001',
    'baidu/ernie-4.5-turbo-128k',
    'baidu/ernie-4.5-turbo-vl-32k',
    'baidu/ernie-x1-turbo-32k',
    'black-forest-labs/flux-kontext-max',
    'black-forest-labs/flux-kontext-max/multi',
    'black-forest-labs/flux-kontext-max/text-to-image',
    'black-forest-labs/flux-kontext-pro',
    'black-forest-labs/flux-kontext-pro/multi',
    'black-forest-labs/flux-kontext-pro/text-to-image',
    'black-forest-labs/flux.1-dev',
    'claude-4-sonnet',
    'deepseek-ai/DeepSeek-R1',
    'deepseek-ai/DeepSeek-R1-0528',
    'deepseek-ai/DeepSeek-V3-0324',
    'deepseek-ai/DeepSeek-V3.1',
    'gemini-2.5-flash',
    'gemini-2.5-flash-image',
    'gemini-2.5-pro',
    'gemini-3.1-flash-image',
    'gemini-3-pro-image',
    'gpt-4.1-mini',
    'grok-4',
    'moonshotai/Kimi-K2-Instruct',
    'openai/gpt-4.1',
    'openai/gpt-5',
    'openai/gpt-5-mini',
    'viduq2',
    'viduq2-pro',
    'viduq2-turbo',
    'viduq2-pro-fast',
    'viduq3-pro',
    'viduq3-turbo',
    'zai-org/glm-5',
    'stepfun-ai/step1x-edit'
]

# Default selected model
DEFAULT_MODEL = "zai-org/glm-5"


class ModelverseChat:
    """OpenAI Chat node with support for text, images, and files"""
    
    # Class-level cache for models list
    _cached_models = None
    _cache_initialized = False
    
    def __init__(self):
        pass
        
    @classmethod
    def get_models_list(cls):
        """Try to fetch models from ModelVerse API, fallback to default models if failed"""
        # Return cached models if already fetched
        if cls._cache_initialized:
            return cls._cached_models
        
        try:
            # Try to create a client and fetch models
            # Note: This will work if there's a valid API key available, otherwise fallback
            client = OpenAI(base_url="https://api.modelverse.cn/v1",api_key="xxxsu")
            models_response = client.models.list()
            
            # Extract model IDs from the response
            if hasattr(models_response, 'data') and models_response.data:
                model_ids = []
                for model in models_response.data:
                    if hasattr(model, 'id') and model.id:
                        model_ids.append(model.id)
                
                if model_ids:
                    print(f"ModelverseChat: Successfully fetched {len(model_ids)} models from API")
                    cls._cached_models = sorted(model_ids)  # Sort for better user experience
                    cls._cache_initialized = True
                    return cls._cached_models
            
            # If no models found, fall back to default
            print("ModelverseChat: No models found in API response, using default models")
            cls._cached_models = DEFAULT_MODELS
            cls._cache_initialized = True
            return cls._cached_models
            
        except Exception as e:
            print(f"ModelverseChat: Failed to fetch models from API ({str(e)}), using default models")
            cls._cached_models = DEFAULT_MODELS
            cls._cache_initialized = True
            return cls._cached_models
    
    @classmethod
    def clear_models_cache(cls):
        """Clear the cached models list to force a refresh on next call"""
        cls._cached_models = None
        cls._cache_initialized = False
        print("ModelverseChat: Models cache cleared")
    
    def display_message_on_node(self, message: str, node_id: str) -> None:
        """Display the current response message on the node UI."""
        render_spec = {
            "node_id": node_id,
            "component": "MessageDisplayWidget",
            "props": {
                "message": message,
            },
        }
        PromptServer.instance.send_sync(
            "display_component",
            render_spec,
        )
    
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get models list dynamically
        available_models = cls.get_models_list()
        
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (available_models, {"default": DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0] if available_models else DEFAULT_MODEL}),
                "user_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "What can you tell me about this?"
                }),
                "temperature": (IO.FLOAT, {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "max_tokens": (IO.INT, {
                    "default": 6000,
                    "min": 1,
                    "max": 128000
                }),
                "top_p": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "system_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "image_in": (IO.IMAGE, {}),
                "files": ("OPENAI_INPUT_FILES", {
                    "default": None,
                    "tooltip": "Optional file(s) to use as context for the model. Accepts inputs from the OpenAI Input Files node."
                }),
                # "response_format": (["text", "json_object"], {"default": "text"}),
                "response_format": (["text"], {"default": "text"}),
                "presence_penalty": (IO.FLOAT, {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "frequency_penalty": (IO.FLOAT, {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("response",)
    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "chat"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Tell ComfyUI that this node's inputs may change"""
        return "static"

    def chat(self, 
             client: Dict[str, str],
             model: str,
             user_prompt: str,
             temperature: float,
             max_tokens: int,
             top_p: float,
             unique_id: Optional[str] = None,
             system_prompt: Optional[str] = "You are a helpful assistant.",
             image_in: Optional[Any] = None,
             files: Optional[List[Any]] = None,
             response_format: str = "text",
             presence_penalty: float = 0.0,
             frequency_penalty: float = 0.0) -> tuple:
        
        # Create ModelverseClient and get API key
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("No API key found in the client")
            
        # Create ModelverseClient instance to get the actual API key
        modelverse_client = ModelverseClient(api_key)
        
        # Initialize OpenAI client with the API key from ModelverseClient
        openai_client = openai.OpenAI(api_key=modelverse_client.api_key,base_url="https://api.modelverse.cn/v1")
        
        # Build messages
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Build user message content
        user_content = []
        
        # Add main prompt
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Add file content if provided
        if files:
            print(f"OpenAIChat: Processing {len(files)} files")
            for file_info in files:
                # Handle the actual file format from OpenAIInputFiles
                if hasattr(file_info, 'file_data') and hasattr(file_info, 'filename'):
                    # Decode base64 content
                    try:
                        # Extract base64 data (remove data:text/plain;base64, prefix)
                        if file_info.file_data.startswith('data:'):
                            base64_data = file_info.file_data.split(',', 1)[1]
                        else:
                            base64_data = file_info.file_data
                        
                        # Decode base64 to text
                        content = base64.b64decode(base64_data).decode('utf-8')
                        
                        file_text = f"\n\nFile: {file_info.filename}\nContent:\n{content}"
                        user_content.append({
                            "type": "text",
                            "text": file_text
                        })
                        print(f"OpenAIChat: Added file {file_info.filename} with {len(content)} characters")
                    except Exception as e:
                        print(f"OpenAIChat: Error decoding file {file_info.filename}: {str(e)}")
                elif isinstance(file_info, dict) and "content" in file_info and "filename" in file_info:
                    # Handle simple dict format (backup)
                    file_text = f"\n\nFile: {file_info['filename']}\nContent:\n{file_info['content']}"
                    user_content.append({
                        "type": "text",
                        "text": file_text
                    })
                    print(f"OpenAIChat: Added file {file_info['filename']} with {len(file_info['content'])} characters")
                else:
                    print(f"OpenAIChat: Invalid file format: {file_info}")
        else:
            print("OpenAIChat: No files provided")
        
        # Add image if provided
        if image_in is not None:
            # Convert tensor to PIL Image
            # Handle both single images and batches
            if len(image_in.shape) == 4:
                # Take first image from batch
                image_in = image_in[0]
            
            # Convert from tensor format (H, W, C) to numpy array
            image_array = image_in.cpu().numpy()
            
            # Ensure values are in 0-255 range
            image_array = np.clip(255. * image_array, 0, 255).astype(np.uint8)
            
            # Create PIL Image
            pil_image = Image.fromarray(image_array)
            
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Add image to content
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                }
            })
        
        # Add user message with all content
        if len(user_content) == 1:
            # If only text, use simple format
            messages.append({
                "role": "user",
                "content": user_content[0]["text"]
            })
        else:
            # If multiple content types, use array format
            messages.append({
                "role": "user",
                "content": user_content
            })
        
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        
        # Add response format if JSON is requested
        if response_format == "json_object":
            api_params["response_format"] = {"type": "json_object"}
        
        try:
            # Make API call to OpenAI
            response = openai_client.chat.completions.create(**api_params)
            
            # Extract response content
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("No content in response")
                
                # Display the response message on the node UI
                if unique_id:
                    self.display_message_on_node(content.strip(), unique_id)
                
                # Return the response as a tuple
                return (content.strip(),)
            else:
                raise ValueError("No choices in response")
                
        except Exception as e:
            error_msg = f"OpenAI API Error: {str(e)}"
            print(error_msg)
            return (error_msg,)


class ModelverseInputFiles:
    """
    Loads and formats input files for OpenAI API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        For details about the supported file input types, see:
        https://platform.openai.com/docs/guides/pdf-files?api-mode=responses
        """
        input_dir = folder_paths.get_input_directory()
        input_files = [
            f
            for f in os.scandir(input_dir)
            if f.is_file()
            and (f.name.endswith(".txt") or f.name.endswith(".pdf") or f.name.endswith(".md"))
            and f.stat().st_size < 32 * 1024 * 1024
        ]
        input_files = sorted(input_files, key=lambda x: x.name)
        input_files = [f.name for f in input_files]
        return {
            "required": {},
            "optional": {
                "file": (
                    IO.COMBO,
                    {
                        "options": input_files,
                        "default": input_files[0] if input_files else None,
                        "tooltip": "Input files to include as context for the model. Only accepts text (.txt), markdown (.md) and PDF (.pdf) files for now.",
                    },
                ),
                "OPENAI_INPUT_FILES": (
                    "OPENAI_INPUT_FILES",
                    {
                        "tooltip": "An optional additional file(s) to batch together with the file loaded from this node. Allows chaining of input files so that a single message can include multiple input files.",
                        "default": None,
                    },
                ),
            },
        }

    DESCRIPTION = "Loads and prepares input files (text, markdown, pdf, etc.) to include as inputs for the OpenAI Chat Node. The files will be read by the OpenAI model when generating a response. 🛈 TIP: Can be chained together with other OpenAI Input File nodes."
    RETURN_TYPES = ("OPENAI_INPUT_FILES",)
    FUNCTION = "prepare_files"
    CATEGORY = "UCLOUD_MODELVERSE"

    def read_file_content(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except:
                return f"Error: Unable to read file {file_path} - encoding issue"
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

    def text_to_data_uri(self, content: str) -> str:
        """Convert text content to data URI."""
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
        return f"data:text/plain;base64,{encoded_content}"

    def create_input_file_content(self, file_path: str):
        """Create a file content object compatible with the reference implementation."""
        content = self.read_file_content(file_path)
        
        # Create a simple object that mimics the expected structure
        class InputFileContent:
            def __init__(self, content, filename):
                self.file_data = self.text_to_data_uri(content)
                self.filename = filename
                self.type = "input_file"
                self.file_id = None
            
            def text_to_data_uri(self, content: str) -> str:
                encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
                return f"data:text/plain;base64,{encoded_content}"
        
        return InputFileContent(content, os.path.basename(file_path))

    def prepare_files(
        self, file: str = None, OPENAI_INPUT_FILES: List[Any] = None
    ) -> tuple:
        """
        Loads and formats input files for OpenAI API.
        """
        files = []
        
        # Add current file if provided
        if file is not None and file != "":
            try:
                file_path = folder_paths.get_annotated_filepath(file)
                input_file_content = self.create_input_file_content(file_path)
                files.append(input_file_content)
                print(f"OpenAIInputFiles: Successfully loaded file {file}")
            except Exception as e:
                print(f"OpenAIInputFiles: Error loading file {file}: {str(e)}")
        
        # Add previous files if provided
        if OPENAI_INPUT_FILES is not None:
            files.extend(OPENAI_INPUT_FILES)
            print(f"OpenAIInputFiles: Added {len(OPENAI_INPUT_FILES)} previous files")
        
        print(f"OpenAIInputFiles: Returning {len(files)} files total")
        return (files,)



# Node registration
NODE_CLASS_MAPPINGS = {
    "ModelverseChat": ModelverseChat,
    "ModelverseInputFiles": ModelverseInputFiles,
    # "OpenAICaptionImage": OpenAICaptionImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelverseChat": "Modelverse Chat",
    "ModelverseInputFiles": "Modelverse Input Files",
    # "OpenAICaptionImage": "OpenAI Caption Image",
}

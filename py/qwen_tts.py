import base64
import io

import requests
import torch
import torchaudio
from comfy.comfy_types.node_typing import IO

BASE_URL = "https://api.modelverse.cn"
SPEECH_PATH = "/v1/audio/speech"

QWEN3_MODEL = "qwen3-tts-flash"
QWEN31_MODELS = ["qwen-audio-3.1-tts-flash", "qwen-audio-3.1-tts-plus"]
MODELS = [QWEN3_MODEL, *QWEN31_MODELS]

LANGUAGE_TYPES = [
    "Auto", "Chinese", "English", "German", "Italian", "Portuguese",
    "Spanish", "Japanese", "Korean", "French", "Russian",
]

FORMATS = ["wav", "mp3", "pcm", "opus"]
SAMPLE_RATES = ["8000", "12000", "16000", "22050", "24000", "44100", "48000"]


def _load_audio(data, sample_rate_hint=None, pcm=False):
    """Decode audio bytes into a ComfyUI AUDIO dict: {"waveform": (1, C, N), "sample_rate": int}."""
    if not data:
        raise ValueError("The TTS API returned empty audio data")

    if pcm:
        waveform = torch.frombuffer(bytearray(data), dtype=torch.int16).float() / 32768.0
        waveform = waveform.unsqueeze(0)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate_hint or 24000)}

    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(data))
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode the returned audio (got {len(data)} bytes): {e}. "
            "Make sure the format is wav/mp3/flac and the audio backend is available."
        ) from e

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


def _error_message(response):
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return error.get("message") or str(error)
        if error:
            return str(error)
        return payload.get("message") or str(payload)
    return str(payload)


class Modelverse_QwenTTS:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": QWEN3_MODEL}),
                "text": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to synthesize. qwen3-tts-flash limits input to 600 characters.",
                }),
                "voice": (IO.STRING, {
                    "default": "Cherry",
                    "tooltip": "qwen3-tts-flash: Cherry / Ethan / Serena / Chelsie; "
                               "qwen-audio-3.1-*: e.g. longanhuan_v3.1",
                }),
            },
            "optional": {
                "language_type": (LANGUAGE_TYPES, {
                    "default": "Auto",
                    "tooltip": "Only used by qwen3-tts-flash (metadata.language_type)",
                }),
                "instructions": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Only used by qwen-audio-3.1-*: emotion/tone/dialect instruction (<=100 chars)",
                }),
                "speed": (IO.FLOAT, {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Only used by qwen-audio-3.1-*",
                }),
                "response_format": (FORMATS, {"default": "wav"}),
                "sample_rate": (SAMPLE_RATES, {"default": "24000"}),
            },
        }

    RETURN_TYPES = (IO.AUDIO, IO.STRING)
    RETURN_NAMES = ("audio", "url")
    FUNCTION = "synthesize"
    CATEGORY = "UCLOUD_MODELVERSE/Audio"

    def synthesize(self, client, model, text, voice, language_type="Auto", instructions="",
                   speed=1.0, response_format="wav", sample_rate="24000"):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set in the client")
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if not voice or not voice.strip():
            raise ValueError("voice must not be empty")

        payload = {"model": model, "input": text, "voice": voice}

        if model == QWEN3_MODEL:
            payload["metadata"] = {"language_type": language_type}
        else:
            if instructions:
                payload["instructions"] = instructions
            payload["speed"] = float(speed)
            payload["response_format"] = response_format
            payload["audio_config"] = {"format": response_format, "sample_rate": int(sample_rate)}

        response = requests.post(
            f"{BASE_URL}{SPEECH_PATH}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )

        if response.status_code != 200:
            raise Exception(f"Qwen TTS request failed ({response.status_code}): {_error_message(response)}")

        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("audio/") or content_type == "application/octet-stream":
            audio_bytes = response.content
            audio_url = ""
        else:
            result = response.json()
            if result.get("error"):
                raise Exception(f"Qwen TTS error: {_error_message(response)}")
            if result.get("code") not in (None, "", 200):
                raise Exception(f"Qwen TTS error: {result.get('message') or result.get('code')}")

            audio_info = (result.get("output") or {}).get("audio") or {}
            audio_url = audio_info.get("url") or ""
            encoded = audio_info.get("data") or ""
            if encoded:
                audio_bytes = base64.b64decode(encoded)
            elif audio_url:
                audio_response = requests.get(audio_url, timeout=300)
                if audio_response.status_code != 200:
                    raise Exception(f"Failed to download synthesized audio ({audio_response.status_code})")
                audio_bytes = audio_response.content
            else:
                raise Exception("Qwen TTS response contains neither audio data nor a URL")

        audio = _load_audio(audio_bytes, sample_rate, pcm=(response_format == "pcm" and model != QWEN3_MODEL))
        return (audio, audio_url)


NODE_CLASS_MAPPINGS = {
    "Modelverse_QwenTTS": Modelverse_QwenTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_QwenTTS": "Modelverse Qwen TTS",
}

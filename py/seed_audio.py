import io

import requests
import torch
import torchaudio
from comfy.comfy_types.node_typing import IO

BASE_URL = "https://api.modelverse.cn"
SPEECH_PATH = "/v1/audio/speech"

MODEL = "seed-audio-1.0"
FORMATS = ["wav", "mp3", "pcm", "ogg_opus"]
SAMPLE_RATES = ["8000", "16000", "24000", "32000", "40000", "44100", "48000"]


def _load_audio(data, sample_rate_hint=None, pcm=False):
    """Decode audio bytes into a ComfyUI AUDIO dict: {"waveform": (1, C, N), "sample_rate": int}."""
    if not data:
        raise ValueError("Seed Audio API returned empty audio data")

    if pcm:
        # Raw 16-bit little-endian PCM stream, mono.
        waveform = torch.frombuffer(bytearray(data), dtype=torch.int16).float() / 32768.0
        waveform = waveform.unsqueeze(0)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate_hint or 40000)}

    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(data))
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode the returned audio (got {len(data)} bytes): {e}. "
            "Try format=wav, or make sure the audio backend supports this container."
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


class Modelverse_SeedAudio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "text_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text or sound description (max 3000 chars). Use @音频1 to reference the speaker audio.",
                }),
            },
            "optional": {
                "speaker": (IO.STRING, {
                    "default": "",
                    "tooltip": "System voice id, e.g. zh_female_vv_uranus_bigtts. Leave empty for pure text-to-audio mode.",
                }),
                "format": (FORMATS, {"default": "wav"}),
                "sample_rate": (SAMPLE_RATES, {"default": "40000"}),
                "speech_rate": (IO.INT, {"default": 0, "min": -50, "max": 100, "step": 1,
                                         "tooltip": "Speech rate, -50 = 0.5x, 100 = 2.0x"}),
                "loudness_rate": (IO.INT, {"default": 0, "min": -50, "max": 100, "step": 1,
                                           "tooltip": "Loudness, -50 = 0.5x, 100 = 2.0x"}),
                "pitch_rate": (IO.INT, {"default": 0, "min": -12, "max": 12, "step": 1,
                                        "tooltip": "Pitch shift, range [-12, 12]"}),
            },
        }

    RETURN_TYPES = (IO.AUDIO,)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "UCLOUD_MODELVERSE/Audio"

    def generate(self, client, text_prompt, speaker="", format="wav", sample_rate="40000",
                 speech_rate=0, loudness_rate=0, pitch_rate=0):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set in the client")
        if not text_prompt or not text_prompt.strip():
            raise ValueError("text_prompt must not be empty")

        payload = {
            "model": MODEL,
            "text_prompt": text_prompt,
            "response_format": format,
            "audio_config": {
                "format": format,
                "sample_rate": int(sample_rate),
                "speech_rate": int(speech_rate),
                "loudness_rate": int(loudness_rate),
                "pitch_rate": int(pitch_rate),
            },
        }
        if speaker and speaker.strip():
            payload["references"] = [{"speaker": speaker.strip()}]

        response = requests.post(
            f"{BASE_URL}{SPEECH_PATH}",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )

        if response.status_code != 200:
            raise Exception(f"Seed Audio request failed ({response.status_code}): {_error_message(response)}")

        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("application/json"):
            raise Exception(f"Seed Audio error: {_error_message(response)}")

        audio = _load_audio(response.content, sample_rate, pcm=(format == "pcm"))
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "Modelverse_SeedAudio": Modelverse_SeedAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_SeedAudio": "Modelverse Seed Audio 1.0",
}

import base64
import io
import re

import requests
import torch
import torchaudio
from comfy.comfy_types.node_typing import IO

from .modelverse_api.client import ModelverseClient

T2A_PATH = "/v1/t2a_v2"

MODELS = ["speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd"]
EMOTIONS = ["", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm", "fluent", "whisper"]
SAMPLE_RATES = ["8000", "16000", "22050", "24000", "32000", "44100"]
BITRATES = ["32000", "64000", "128000", "256000"]
FORMATS = ["mp3", "pcm", "flac", "wav"]
CHANNELS = ["1", "2"]
OUTPUT_FORMATS = ["hex", "url"]


def _load_audio(data, sample_rate_hint=None, pcm=False):
    """Decode audio bytes into a ComfyUI AUDIO dict: {"waveform": (1, C, N), "sample_rate": int}."""
    if not data:
        raise ValueError("MiniMax Speech returned empty audio data")

    if pcm:
        # Raw 16-bit little-endian PCM stream.
        waveform = torch.frombuffer(bytearray(data), dtype=torch.int16).float() / 32768.0
        waveform = waveform.unsqueeze(0)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate_hint or 32000)}

    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(data))
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode the returned audio (got {len(data)} bytes): {e}. "
            "Make sure the audio backend supports this container; otherwise use format=mp3 or wav."
        ) from e

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


def _decode_text_audio(text, sample_rate_hint=None, pcm=False):
    """MiniMax returns audio either hex-encoded or base64-encoded; try both."""
    text = text.strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]

    candidates = []
    if re.fullmatch(r"[0-9a-fA-F]+", text) and len(text) % 2 == 0:
        candidates.append(bytes.fromhex(text))
    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", text):
        try:
            candidates.append(base64.b64decode(text))
        except Exception:
            pass

    if not candidates:
        raise ValueError("Unrecognized audio encoding in MiniMax Speech response (expected hex or base64)")

    last_error = None
    for data in candidates:
        try:
            return _load_audio(data, sample_rate_hint, pcm)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to decode audio from MiniMax Speech response: {last_error}")


class Modelverse_MiniMaxSpeech:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (MODELS, {"default": "speech-2.8-hd"}),
                "text": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to synthesize (max 9999 chars); use <#0.5#> for pauses",
                }),
                "voice_id": (IO.STRING, {
                    "default": "male-qn-qingse",
                    "tooltip": "System, cloned or text-generated voice id",
                }),
            },
            "optional": {
                "speed": (IO.FLOAT, {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "vol": (IO.FLOAT, {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "pitch": (IO.INT, {"default": 0, "min": -12, "max": 12, "step": 1}),
                "emotion": (EMOTIONS, {"default": "", "tooltip": "Leave empty to let the model decide"}),
                "format": (FORMATS, {"default": "mp3"}),
                "sample_rate": (SAMPLE_RATES, {"default": "32000"}),
                "bitrate": (BITRATES, {"default": "128000"}),
                "channel": (CHANNELS, {"default": "1"}),
                "output_format": (OUTPUT_FORMATS, {"default": "hex",
                                                   "tooltip": "hex: audio bytes in the response; url: downloadable link (24h)"}),
                "subtitle_enable": (IO.BOOLEAN, {"default": False}),
            },
        }

    RETURN_TYPES = (IO.AUDIO, IO.STRING)
    RETURN_NAMES = ("audio", "url")
    FUNCTION = "synthesize"
    CATEGORY = "UCLOUD_MODELVERSE/Audio"

    def synthesize(self, client, model, text, voice_id, speed=1.0, vol=1.0, pitch=0,
                   emotion="", format="mp3", sample_rate="32000", bitrate="128000",
                   channel="1", output_format="hex", subtitle_enable=False):
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("API key is not set in the client")
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if not voice_id or not voice_id.strip():
            raise ValueError("voice_id must not be empty")

        voice_setting = {
            "voice_id": voice_id,
            "speed": float(speed),
            "vol": float(vol),
            "pitch": int(pitch),
        }
        if emotion:
            voice_setting["emotion"] = emotion

        payload = {
            "model": model,
            "text": text,
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": {
                "sample_rate": int(sample_rate),
                "bitrate": int(bitrate),
                "format": format,
                "channel": int(channel),
            },
            "subtitle_enable": bool(subtitle_enable),
            "output_format": output_format,
        }

        mv_client = ModelverseClient(api_key)
        result = mv_client.post(T2A_PATH, payload, timeout=600)

        base_resp = result.get("base_resp") or {}
        if base_resp.get("status_code", 0) != 0:
            raise Exception(
                f"MiniMax Speech error ({base_resp.get('status_code')}): {base_resp.get('status_msg', 'Unknown error')}"
            )

        audio_text = (result.get("data") or {}).get("audio") or ""
        if not audio_text:
            raise Exception("MiniMax Speech response does not contain audio data")

        extra_info = result.get("extra_info") or {}
        sample_rate_hint = extra_info.get("audio_sample_rate") or sample_rate

        audio_url = ""
        if isinstance(audio_text, str) and audio_text.startswith(("http://", "https://")):
            audio_url = audio_text
            download = requests.get(audio_url, timeout=300)
            if download.status_code != 200:
                raise Exception(f"Failed to download MiniMax audio ({download.status_code})")
            audio = _load_audio(download.content, sample_rate_hint, pcm=(format == "pcm"))
        else:
            audio = _decode_text_audio(audio_text, sample_rate_hint, pcm=(format == "pcm"))

        return (audio, audio_url)


NODE_CLASS_MAPPINGS = {
    "Modelverse_MiniMaxSpeech": Modelverse_MiniMaxSpeech,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse_MiniMaxSpeech": "Modelverse MiniMax Speech",
}

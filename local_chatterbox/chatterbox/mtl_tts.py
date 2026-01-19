"""
Chatterbox Multilingual TTS - Text-to-speech supporting 23 languages.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

# Optional Perth watermarking - gracefully handle import failure
try:
    import perth
    PERTH_AVAILABLE = True
except ImportError:
    PERTH_AVAILABLE = False
    print("Warning: Perth watermarking not available. Audio will be generated without watermarking.")

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .paths import get_chatterbox_multilingual_dir, download_to_local


REPO_ID = "ResembleAI/chatterbox"
MTL_MODEL_FILES = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """Punctuation normalization for multilingual model."""
    if len(text) == 0:
        return "You need to add some text for me to talk."

    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """Conditionals for T3 and S3Gen."""
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    """
    Chatterbox Multilingual TTS - Text-to-speech supporting 23 languages.

    Supported languages:
    - Arabic (ar), Danish (da), German (de), Greek (el), English (en)
    - Spanish (es), Finnish (fi), French (fr), Hebrew (he), Hindi (hi)
    - Italian (it), Japanese (ja), Korean (ko), Malay (ms), Dutch (nl)
    - Norwegian (no), Polish (pl), Portuguese (pt), Russian (ru)
    - Swedish (sv), Swahili (sw), Turkish (tr), Chinese (zh)
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker() if PERTH_AVAILABLE else None

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True, map_location=map_location))
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True, map_location=map_location))
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxMultilingualTTS':
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+.")
            device = "cpu"

        # Download models to centralized location
        local_dir = get_chatterbox_multilingual_dir()
        print(f"[FL Chatterbox Multilingual] Model download path: {local_dir}")
        download_to_local(REPO_ID, MTL_MODEL_FILES, local_dir)

        return cls.from_local(local_dir, device)

    def _trim_trailing_silence(self, wav, threshold_db=-40, min_silence_duration=0.5):
        """
        Trim trailing silence/noise from audio.

        Args:
            wav: Audio waveform as numpy array
            threshold_db: Silence threshold in dB (default -40dB)
            min_silence_duration: Minimum silence duration to trigger trimming (seconds)

        Returns:
            Trimmed audio waveform
        """
        # Convert threshold from dB to linear amplitude
        threshold = 10 ** (threshold_db / 20)

        # Calculate RMS energy in windows
        window_size = int(0.02 * self.sr)  # 20ms windows
        hop_size = window_size // 2

        # Compute RMS for each window
        num_windows = (len(wav) - window_size) // hop_size + 1
        if num_windows <= 0:
            return wav

        rms = np.zeros(num_windows)
        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size
            rms[i] = np.sqrt(np.mean(wav[start:end] ** 2))

        # Find the last window above threshold
        above_threshold = rms > threshold
        if not np.any(above_threshold):
            # All silence, return minimal audio
            return wav[:int(0.1 * self.sr)]

        last_voice_idx = np.where(above_threshold)[0][-1]

        # Convert back to sample index and add a small buffer
        trim_sample = (last_voice_idx + 1) * hop_size + window_size
        trim_sample = min(trim_sample + int(0.1 * self.sr), len(wav))  # Add 100ms buffer

        return wav[:trim_sample]

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """Prepare voice conditionals from reference audio."""
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        """
        Generate speech from text in specified language.

        Args:
            text: Text to speak
            language_id: Two-letter language code (e.g., 'en', 'fr', 'ja', 'zh')
            audio_prompt_path: Path to reference voice audio (min 6 seconds)
            exaggeration: Emotion intensity (0.0-1.0+, default 0.5)
            cfg_weight: Classifier-free guidance weight (default 0.5)
            temperature: Sampling temperature (default 0.8)
            repetition_penalty: Penalty for token repetition (default 2.0)
            min_p: Minimum probability threshold (default 0.05)
            top_p: Nucleus sampling threshold (default 1.0)

        Returns:
            Tensor of audio waveform
        """
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()

            # Trim trailing silence/noise - workaround for known multilingual model issue
            # See: https://github.com/resemble-ai/chatterbox/issues/287
            wav = self._trim_trailing_silence(wav)

            if self.watermarker is not None:
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                return torch.from_numpy(watermarked_wav).unsqueeze(0)
            else:
                return torch.from_numpy(wav).unsqueeze(0)

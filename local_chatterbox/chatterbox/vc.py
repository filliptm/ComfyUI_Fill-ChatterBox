from pathlib import Path

import numpy as np
import librosa
import torch

# Optional Perth watermarking - gracefully handle import failure
try:
    import perth
    PERTH_AVAILABLE = True
except ImportError:
    PERTH_AVAILABLE = False
    print("Warning: Perth watermarking not available. Audio will be generated without watermarking.")

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .paths import get_chatterbox_tts_dir, download_to_local


REPO_ID = "ResembleAI/chatterbox"
VC_MODEL_FILES = ["s3gen.pt", "conds.pt"]


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker() if PERTH_AVAILABLE else None
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", map_location=map_location)
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        # Download models to centralized location: ComfyUI/models/chatterbox/chatterbox/
        local_dir = get_chatterbox_tts_dir()
        print(f"[FL Chatterbox VC] Model download path: {local_dir}")
        download_to_local(REPO_ID, VC_MODEL_FILES, local_dir)

        return cls.from_local(local_dir, device)

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        # Ensure float32 for PyTorch compatibility
        s3gen_ref_wav = s3gen_ref_wav.astype(np.float32)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        with torch.inference_mode():
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            if self.watermarker is not None:
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                return torch.from_numpy(watermarked_wav).unsqueeze(0)
            else:
                return torch.from_numpy(wav).unsqueeze(0)
import os
import torch
import torchaudio
import tempfile

from .local_chatterbox.chatterbox.tts import ChatterboxTTS
from comfy.utils import ProgressBar

class FL_ChatterboxDialogTTSNode:
    """
    TTS Node that accepts dialog with speaker labels and generates audio using separate voice prompts.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialog_text": ("STRING", {"multiline": True, "default": "SPEAKER A: Test test\nSPEAKER B: 1 2 3"}),
                "speaker_A_Audio": ("AUDIO",),
                "speaker_B_Audio": ("AUDIO",),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "speaker_C_Audio": ("AUDIO",),
                "speaker_D_Audio": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("dialog_audio", "speaker_a_audio", "speaker_b_audio", "speaker_c_audio", "speaker_d_audio", "message")
    FUNCTION = "generate_dialog"
    CATEGORY = "ChatterBox"

    _model = None
    _device = None

    def generate_dialog(self, dialog_text, speaker_A_Audio, speaker_B_Audio,
                        exaggeration, cfg_weight, temperature, seed,
                        speaker_C_Audio=None, speaker_D_Audio=None,
                        use_cpu=False, keep_model_loaded=False):
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        
        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        pbar = ProgressBar(100)
        message = f"Running on {device}"

        def save_temp_audio(audio_data):
            path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            torchaudio.save(path, audio_data['waveform'].squeeze(0), audio_data['sample_rate'])
            return path

        prompt_a_path = save_temp_audio(speaker_A_Audio)
        prompt_b_path = save_temp_audio(speaker_B_Audio)
        temp_files = [prompt_a_path, prompt_b_path]
        
        # Handle optional speakers C and D
        prompt_c_path = None
        prompt_d_path = None
        if speaker_C_Audio is not None:
            prompt_c_path = save_temp_audio(speaker_C_Audio)
            temp_files.append(prompt_c_path)
        if speaker_D_Audio is not None:
            prompt_d_path = save_temp_audio(speaker_D_Audio)
            temp_files.append(prompt_d_path)

        if self._model is None or self._device != device:
            self._model = ChatterboxTTS.from_pretrained(device=device)
            self._device = device
        tts = self._model

        lines = dialog_text.strip().splitlines()
        speaker_a_waveforms = []
        speaker_b_waveforms = []
        speaker_c_waveforms = []
        speaker_d_waveforms = []
        combined_dialog_waveforms = []

        for i, line in enumerate(lines):
            wav = None
            if line.startswith("SPEAKER A:"):
                content = line[len("SPEAKER A:"):].strip()
                prompt_path = prompt_a_path
                pbar.update_absolute(int((i / len(lines)) * 80))
                current_speaker_wav = tts.generate(
                    text=content,
                    audio_prompt_path=prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
                speaker_a_waveforms.append(current_speaker_wav)
                combined_dialog_waveforms.append(current_speaker_wav)
                # Add silence to other speakers' tracks
                silence = torch.zeros_like(current_speaker_wav)
                speaker_b_waveforms.append(silence)
                speaker_c_waveforms.append(silence)
                speaker_d_waveforms.append(silence)
            elif line.startswith("SPEAKER B:"):
                content = line[len("SPEAKER B:"):].strip()
                prompt_path = prompt_b_path
                pbar.update_absolute(int((i / len(lines)) * 80))
                current_speaker_wav = tts.generate(
                    text=content,
                    audio_prompt_path=prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
                speaker_b_waveforms.append(current_speaker_wav)
                combined_dialog_waveforms.append(current_speaker_wav)
                # Add silence to other speakers' tracks
                silence = torch.zeros_like(current_speaker_wav)
                speaker_a_waveforms.append(silence)
                speaker_c_waveforms.append(silence)
                speaker_d_waveforms.append(silence)
            elif line.startswith("SPEAKER C:") and prompt_c_path is not None:
                content = line[len("SPEAKER C:"):].strip()
                prompt_path = prompt_c_path
                pbar.update_absolute(int((i / len(lines)) * 80))
                current_speaker_wav = tts.generate(
                    text=content,
                    audio_prompt_path=prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
                speaker_c_waveforms.append(current_speaker_wav)
                combined_dialog_waveforms.append(current_speaker_wav)
                # Add silence to other speakers' tracks
                silence = torch.zeros_like(current_speaker_wav)
                speaker_a_waveforms.append(silence)
                speaker_b_waveforms.append(silence)
                speaker_d_waveforms.append(silence)
            elif line.startswith("SPEAKER D:") and prompt_d_path is not None:
                content = line[len("SPEAKER D:"):].strip()
                prompt_path = prompt_d_path
                pbar.update_absolute(int((i / len(lines)) * 80))
                current_speaker_wav = tts.generate(
                    text=content,
                    audio_prompt_path=prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
                speaker_d_waveforms.append(current_speaker_wav)
                combined_dialog_waveforms.append(current_speaker_wav)
                # Add silence to other speakers' tracks
                silence = torch.zeros_like(current_speaker_wav)
                speaker_a_waveforms.append(silence)
                speaker_b_waveforms.append(silence)
                speaker_c_waveforms.append(silence)
            else:
                continue  # skip malformed line or missing prompt

        if not combined_dialog_waveforms:
            empty_audio = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": tts.sr if tts else 16000}
            return (empty_audio, empty_audio, empty_audio, empty_audio, empty_audio, "No valid dialog lines found.")

        combined_waveform = torch.cat(combined_dialog_waveforms, dim=-1)
        speaker_a_track = torch.cat(speaker_a_waveforms, dim=-1)
        speaker_b_track = torch.cat(speaker_b_waveforms, dim=-1)
        speaker_c_track = torch.cat(speaker_c_waveforms, dim=-1)
        speaker_d_track = torch.cat(speaker_d_waveforms, dim=-1)

        dialog_audio = {"waveform": combined_waveform.unsqueeze(0), "sample_rate": tts.sr}
        speaker_a_audio = {"waveform": speaker_a_track.unsqueeze(0), "sample_rate": tts.sr}
        speaker_b_audio = {"waveform": speaker_b_track.unsqueeze(0), "sample_rate": tts.sr}
        speaker_c_audio = {"waveform": speaker_c_track.unsqueeze(0), "sample_rate": tts.sr}
        speaker_d_audio = {"waveform": speaker_d_track.unsqueeze(0), "sample_rate": tts.sr}

        for f in temp_files:
            os.unlink(f)

        pbar.update_absolute(100)
        return (dialog_audio, speaker_a_audio, speaker_b_audio, speaker_c_audio, speaker_d_audio, "Dialog synthesized successfully.")

NODE_CLASS_MAPPINGS = {
    "FL_ChatterboxDialogTTS": FL_ChatterboxDialogTTSNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ChatterboxDialogTTS": "FL Chatterbox Dialog TTS",
}
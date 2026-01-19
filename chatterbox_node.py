import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import scipy.io.wavfile as wavfile


def save_audio_wav(filepath: str, waveform: torch.Tensor, sample_rate: int):
    """
    Save audio to WAV file with fallback for TorchCodec issues.

    Args:
        filepath: Path to save the audio file
        waveform: Audio tensor of shape (channels, samples) or (samples,)
        sample_rate: Sample rate in Hz
    """
    # Ensure waveform is float32 for consistency
    waveform = waveform.float()

    try:
        # Try torchaudio first with explicit backend
        torchaudio.save(filepath, waveform, sample_rate, backend="soundfile")
    except Exception:
        try:
            # Fallback: try without specifying backend
            torchaudio.save(filepath, waveform, sample_rate)
        except Exception:
            # Final fallback: use scipy
            audio_np = waveform.cpu().numpy().astype(np.float32)
            # scipy expects (samples, channels) for stereo, or (samples,) for mono
            if len(audio_np.shape) == 2:
                audio_np = audio_np.T  # Transpose from (channels, samples) to (samples, channels)
            # Normalize to int16 range for WAV file
            audio_np = (audio_np * 32767).astype(np.int16)
            wavfile.write(filepath, sample_rate, audio_np)


# Import from the local chatterbox implementation
from .local_chatterbox.chatterbox import ChatterboxTTS
from .local_chatterbox.chatterbox import ChatterboxTurboTTS
from .local_chatterbox.chatterbox import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .local_chatterbox.chatterbox import ChatterboxVC

from comfy.utils import ProgressBar

# ============================================================================
# Global model cache - persists across node executions
# Using module-level globals instead of class variables for reliability
# ============================================================================
_MODEL_CACHE: Dict[str, Any] = {}


def get_cached_model(model_type: str, device: str):
    """Get a cached model if available and on correct device."""
    cache_key = f"{model_type}_{device}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        print(f"[FL Chatterbox] Using cached {model_type} model on {device}")
        return cached
    return None


def cache_model(model_type: str, device: str, model):
    """Store a model in the cache."""
    cache_key = f"{model_type}_{device}"
    _MODEL_CACHE[cache_key] = model
    print(f"[FL Chatterbox] Cached {model_type} model on {device}")


def clear_cached_model(model_type: str = None):
    """Clear cached model(s). If model_type is None, clear all."""
    global _MODEL_CACHE
    if model_type is None:
        _MODEL_CACHE.clear()
        print("[FL Chatterbox] Cleared all cached models")
    else:
        keys_to_remove = [k for k in _MODEL_CACHE if k.startswith(f"{model_type}_")]
        for key in keys_to_remove:
            del _MODEL_CACHE[key]
        if keys_to_remove:
            print(f"[FL Chatterbox] Cleared cached {model_type} model(s)")

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ============================================================================
# Centralized model path management
# ============================================================================
def get_chatterbox_models_dir() -> Path:
    """
    Get the centralized models directory for all Chatterbox models.
    Returns: Path to ComfyUI/models/chatterbox/
    """
    current_dir = Path(__file__).parent
    comfyui_root = current_dir.parent.parent  # custom_nodes -> ComfyUI

    models_dir = comfyui_root / "models" / "chatterbox"

    # Verify we're in a valid ComfyUI structure
    if not (comfyui_root / "custom_nodes").exists():
        models_dir = current_dir / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_chatterbox_models(repo_id: str, filenames: list, local_dir: Path) -> Path:
    """Download model files from HuggingFace to our centralized location."""
    from huggingface_hub import hf_hub_download
    import shutil

    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        local_path = local_dir / filename
        if not local_path.exists():
            print(f"[FL Chatterbox] Downloading {filename}...")
            try:
                cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
                shutil.copy2(cached_path, local_path)
            except Exception as e:
                print(f"[FL Chatterbox] Error downloading {filename}: {e}")
                raise
        else:
            print(f"[FL Chatterbox] Using cached {filename}")

    return local_dir


def load_turbo_model(device: str) -> ChatterboxTurboTTS:
    """Load Turbo TTS model from centralized path."""
    # Check MPS availability
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("[FL Chatterbox] MPS not available, falling back to CPU")

    local_dir = get_chatterbox_models_dir() / "chatterbox_turbo"
    print(f"[FL Chatterbox Turbo] Model download path: {local_dir}")

    # Files needed for Turbo model
    turbo_files = [
        "ve.safetensors",
        "t3_turbo_v1.safetensors",
        "s3gen_meanflow.safetensors",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "conds.pt",
    ]

    download_chatterbox_models("ResembleAI/chatterbox-turbo", turbo_files, local_dir)
    return ChatterboxTurboTTS.from_local(str(local_dir), device)


def load_tts_model(device: str) -> ChatterboxTTS:
    """Load standard TTS model from centralized path."""
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("[FL Chatterbox] MPS not available, falling back to CPU")

    local_dir = get_chatterbox_models_dir() / "chatterbox"
    print(f"[FL Chatterbox] Model download path: {local_dir}")

    tts_files = [
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "conds.pt",
    ]

    download_chatterbox_models("ResembleAI/chatterbox", tts_files, local_dir)
    return ChatterboxTTS.from_local(str(local_dir), device)


def load_multilingual_model(device: str) -> ChatterboxMultilingualTTS:
    """Load Multilingual TTS model from centralized path."""
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("[FL Chatterbox] MPS not available, falling back to CPU")

    local_dir = get_chatterbox_models_dir() / "chatterbox_multilingual"
    print(f"[FL Chatterbox Multilingual] Model download path: {local_dir}")

    mtl_files = [
        "ve.pt",
        "t3_mtl23ls_v2.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
        "Cangjie5_TC.json",
    ]

    download_chatterbox_models("ResembleAI/chatterbox", mtl_files, local_dir)
    return ChatterboxMultilingualTTS.from_local(str(local_dir), device)


def load_vc_model(device: str) -> ChatterboxVC:
    """Load Voice Conversion model from centralized path."""
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("[FL Chatterbox] MPS not available, falling back to CPU")

    local_dir = get_chatterbox_models_dir() / "chatterbox_vc"
    print(f"[FL Chatterbox VC] Model download path: {local_dir}")

    # VC model requires s3gen.pt (not safetensors) - see vc.py VC_MODEL_FILES
    vc_files = [
        "s3gen.pt",
        "conds.pt",
    ]

    download_chatterbox_models("ResembleAI/chatterbox", vc_files, local_dir)
    return ChatterboxVC.from_local(str(local_dir), device)

# Monkey patch torch.load to use MPS or CPU if map_location is not specified
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        # Determine the appropriate device (MPS for Mac, else CPU)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        kwargs['map_location'] = torch.device(device)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


class AudioNodeBase:
    """Base class for audio nodes with common utilities."""
    
    @staticmethod
    def create_empty_tensor(audio, frame_rate, height, width, channels=None):
        """Create an empty tensor with dimensions based on audio duration."""
        audio_duration = audio['waveform'].shape[-1] / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)
        if channels is None:
            return torch.zeros((num_frames, height, width), dtype=torch.float32)
        else:
            return torch.zeros((num_frames, height, width, channels), dtype=torch.float32)

# Text-to-Speech node
class FL_ChatterboxTTSNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Text-to-Speech functionality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"
    
    def generate_speech(self, text, exaggeration, cfg_weight, temperature, seed, audio_prompt=None, use_cpu=False, keep_model_loaded=False):
        """
        Generate speech from text.
        
        Args:
            text: The text to convert to speech.
            exaggeration: Controls emotion intensity (0.25-2.0).
            cfg_weight: Controls pace/classifier-free guidance (0.2-1.0).
            temperature: Controls randomness in generation (0.05-5.0).
            seed: Random seed for reproducible generation.
            audio_prompt: AUDIO object containing the reference voice for TTS voice cloning.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after generation.
            
        Returns:
            Tuple of (audio, message)
        """
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
        # Determine device to use
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif torch.backends.mps.is_available() and device == "mps":
             message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
             message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference" # Should be CPU if no GPU found
        
        # Create temporary files for any audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the audio prompt if provided
        audio_prompt_path = None
        if audio_prompt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_prompt:
                    audio_prompt_path = temp_prompt.name
                    temp_files.append(audio_prompt_path)
                
                # Save the audio prompt to the temporary file
                prompt_waveform = audio_prompt['waveform'].squeeze(0)
                save_audio_wav(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
                message += f"\nUsing provided audio prompt for voice cloning: {audio_prompt_path}"
                
                # Debug: Check if the file exists and has content
                if os.path.exists(audio_prompt_path):
                    file_size = os.path.getsize(audio_prompt_path)
                    message += f"\nAudio prompt file created successfully: {file_size} bytes"
                else:
                    message += f"\nWarning: Audio prompt file was not created properly"
            except Exception as e:
                message += f"\nError creating audio prompt file: {str(e)}"
                audio_prompt_path = None
        
        tts_model = None
        wav = None # Initialize wav to None
        audio_data = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000} # Initialize with empty audio
        pbar = ProgressBar(100) # Simple progress bar for overall process
        try:
            # Load the TTS model or reuse if cached
            tts_model = get_cached_model("tts", device)
            if tts_model is not None:
                message += f"\nReusing loaded TTS model on {device}..."
            else:
                # Clear any cached model on different device
                clear_cached_model("tts")

                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started
                tts_model = load_tts_model(device=device)
                pbar.update_absolute(50) # Indicate model loading finished

                if keep_model_loaded:
                    cache_model("tts", device, tts_model)
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."

            # Generate speech
            message += f"\nGenerating speech for: {text[:50]}..." if len(text) > 50 else f"\nGenerating speech for: {text}"
            if audio_prompt_path:
                message += f"\nUsing audio prompt: {audio_prompt_path}"
            
            pbar.update_absolute(60) # Indicate generation started
            wav = tts_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            pbar.update_absolute(90) # Indicate generation finished
            
            audio_data = {
                "waveform": wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": tts_model.sr
            }
            message += f"\nSpeech generated successfully"
            return (audio_data, message)
            
        except RuntimeError as e:
            # Check for CUDA or MPS errors and attempt fallback to CPU
            error_str = str(e)
            fallback_to_cpu = False
            if "CUDA" in error_str and device == "cuda":
                message += "\nCUDA error detected during TTS. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                 message += "\nMPS error detected during TTS. Falling back to CPU..."
                 fallback_to_cpu = True

            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model
                clear_cached_model("tts")

                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started (fallback)
                tts_model = load_tts_model(device=device)
                pbar.update_absolute(50) # Indicate model loading finished (fallback)
                # Note: keep_model_loaded logic is applied after successful generation
                # to avoid keeping a failed model loaded.

                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                pbar.update_absolute(90) # Indicate generation finished (fallback)
                audio_data = {
                    "waveform": wav.unsqueeze(0),  # Add batch dimension
                    "sample_rate": tts_model.sr
                }
                message += f"\nSpeech generated successfully after fallback."
                return (audio_data, message)
            else:
                message += f"\nError during TTS: {str(e)}"
                return (audio_data, message)
        except Exception as e:
             message += f"\nAn unexpected error occurred during TTS: {str(e)}"
             return (audio_data, message)
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            # If keep_model_loaded is False, clear the cache
            if not keep_model_loaded:
                clear_cached_model("tts")

        pbar.update_absolute(100) # Ensure progress bar completes on success or error
        return (audio_data, message) # Fallback return, should ideally not be reached

# Turbo Text-to-Speech node
class FL_ChatterboxTurboTTSNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Turbo Text-to-Speech functionality.

    Faster GPT2-based TTS with paralinguistic tag support.
    Supports tags like: [laugh], [sigh], [gasp], [chuckle], [cough], [sniff], [groan], [shush], [clear throat]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test. [laugh] Isn't that funny?"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 1000, "min": 1, "max": 5000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 3.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"

    def generate_speech(self, text, temperature, top_k, top_p, repetition_penalty, seed, audio_prompt=None, use_cpu=False, keep_model_loaded=False):
        """
        Generate speech from text using Turbo model.

        Args:
            text: The text to convert to speech. Supports paralinguistic tags like [laugh], [sigh], etc.
            temperature: Controls randomness in generation (0.05-2.0).
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling threshold (0.1-1.0).
            repetition_penalty: Penalty for token repetition (1.0-3.0).
            seed: Random seed for reproducible generation.
            audio_prompt: AUDIO object containing the reference voice for TTS voice cloning (min 5 seconds).
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after generation.

        Returns:
            Tuple of (audio, message)
        """
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        import random
        np.random.seed(seed)
        random.seed(seed)

        # Determine device to use
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif torch.backends.mps.is_available() and device == "mps":
            message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
            message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference"

        # Create temporary files for any audio inputs
        import tempfile
        temp_files = []

        # Create a temporary file for the audio prompt if provided
        audio_prompt_path = None
        if audio_prompt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_prompt:
                    audio_prompt_path = temp_prompt.name
                    temp_files.append(audio_prompt_path)

                # Save the audio prompt to the temporary file
                prompt_waveform = audio_prompt['waveform'].squeeze(0)
                save_audio_wav(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
                message += f"\nUsing provided audio prompt for voice cloning"

                # Check audio duration (Turbo requires min 5 seconds)
                duration = prompt_waveform.shape[-1] / audio_prompt['sample_rate']
                if duration < 5.0:
                    message += f"\nWarning: Audio prompt is {duration:.1f}s, Turbo model requires at least 5 seconds"
            except Exception as e:
                message += f"\nError creating audio prompt file: {str(e)}"
                audio_prompt_path = None

        turbo_model = None
        wav = None
        audio_data = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 24000}
        pbar = ProgressBar(100)

        try:
            # Load the Turbo model or reuse if cached
            turbo_model = get_cached_model("turbo", device)
            if turbo_model is not None:
                message += f"\nReusing loaded Turbo TTS model on {device}..."
            else:
                # Clear any cached model on different device
                clear_cached_model("turbo")

                message += f"\nLoading Turbo TTS model on {device}..."
                pbar.update_absolute(10)
                turbo_model = load_turbo_model(device=device)
                pbar.update_absolute(50)

                if keep_model_loaded:
                    cache_model("turbo", device, turbo_model)
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."

            # Generate speech
            message += f"\nGenerating speech for: {text[:50]}..." if len(text) > 50 else f"\nGenerating speech for: {text}"

            pbar.update_absolute(60)
            wav = turbo_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            pbar.update_absolute(90)

            audio_data = {
                "waveform": wav.unsqueeze(0),
                "sample_rate": turbo_model.sr
            }
            message += f"\nSpeech generated successfully (Turbo)"
            return (audio_data, message)

        except RuntimeError as e:
            error_str = str(e)
            fallback_to_cpu = False
            if "CUDA" in error_str and device == "cuda":
                message += "\nCUDA error detected. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                message += "\nMPS error detected. Falling back to CPU..."
                fallback_to_cpu = True

            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model
                clear_cached_model("turbo")

                message += f"\nLoading Turbo TTS model on CPU..."
                pbar.update_absolute(10)
                turbo_model = load_turbo_model(device=device)
                pbar.update_absolute(50)

                wav = turbo_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                pbar.update_absolute(90)

                audio_data = {
                    "waveform": wav.unsqueeze(0),
                    "sample_rate": turbo_model.sr
                }
                message += f"\nSpeech generated successfully after fallback (Turbo)"
                return (audio_data, message)
            else:
                message += f"\nError during Turbo TTS: {str(e)}"
                return (audio_data, message)
        except Exception as e:
            message += f"\nAn unexpected error occurred during Turbo TTS: {str(e)}"
            return (audio_data, message)
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            # If keep_model_loaded is False, clear the cache
            if not keep_model_loaded:
                clear_cached_model("turbo")

        pbar.update_absolute(100)
        return (audio_data, message)


# Multilingual Text-to-Speech node
class FL_ChatterboxMultilingualTTSNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Multilingual Text-to-Speech functionality.

    Supports 23 languages: Arabic, Danish, German, Greek, English, Spanish, Finnish,
    French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian,
    Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Create language choices from SUPPORTED_LANGUAGES
        language_choices = [(code, f"{name} ({code})") for code, name in SUPPORTED_LANGUAGES.items()]
        language_list = [f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES.items()]

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a multilingual test."}),
                "language": (language_list, {"default": "English (en)"}),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 2.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"

    def generate_speech(self, text, language, exaggeration, cfg_weight, temperature, repetition_penalty, min_p, top_p, seed, audio_prompt=None, use_cpu=False, keep_model_loaded=False):
        """
        Generate speech from text in specified language.

        Args:
            text: The text to convert to speech.
            language: The language for speech generation (e.g., "English (en)").
            exaggeration: Controls emotion intensity (0.0-2.0).
            cfg_weight: Classifier-free guidance weight (0.0-1.0).
            temperature: Controls randomness in generation (0.05-2.0).
            repetition_penalty: Penalty for token repetition (1.0-5.0).
            min_p: Minimum probability threshold (0.0-1.0).
            top_p: Nucleus sampling threshold (0.1-1.0).
            seed: Random seed for reproducible generation.
            audio_prompt: AUDIO object containing the reference voice (min 6 seconds).
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after generation.

        Returns:
            Tuple of (audio, message)
        """
        # Extract language code from selection (e.g., "English (en)" -> "en")
        language_id = language.split("(")[-1].replace(")", "").strip()

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        import random
        np.random.seed(seed)
        random.seed(seed)

        # Determine device to use
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif torch.backends.mps.is_available() and device == "mps":
            message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
            message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference"

        message += f"\nLanguage: {language}"

        # Create temporary files for any audio inputs
        import tempfile
        temp_files = []

        # Create a temporary file for the audio prompt if provided
        audio_prompt_path = None
        if audio_prompt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_prompt:
                    audio_prompt_path = temp_prompt.name
                    temp_files.append(audio_prompt_path)

                # Save the audio prompt to the temporary file
                prompt_waveform = audio_prompt['waveform'].squeeze(0)
                save_audio_wav(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
                message += f"\nUsing provided audio prompt for voice cloning"

                # Check audio duration (Multilingual requires min 6 seconds)
                duration = prompt_waveform.shape[-1] / audio_prompt['sample_rate']
                if duration < 6.0:
                    message += f"\nWarning: Audio prompt is {duration:.1f}s, Multilingual model requires at least 6 seconds"
            except Exception as e:
                message += f"\nError creating audio prompt file: {str(e)}"
                audio_prompt_path = None

        mtl_model = None
        wav = None
        audio_data = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 24000}
        pbar = ProgressBar(100)

        try:
            # Load the Multilingual model or reuse if cached
            mtl_model = get_cached_model("multilingual", device)
            if mtl_model is not None:
                message += f"\nReusing loaded Multilingual TTS model on {device}..."
            else:
                # Clear any cached model on different device
                clear_cached_model("multilingual")

                message += f"\nLoading Multilingual TTS model on {device}..."
                pbar.update_absolute(10)
                mtl_model = load_multilingual_model(device=device)
                pbar.update_absolute(50)

                if keep_model_loaded:
                    cache_model("multilingual", device, mtl_model)
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."

            # Generate speech
            message += f"\nGenerating speech for: {text[:50]}..." if len(text) > 50 else f"\nGenerating speech for: {text}"

            pbar.update_absolute(60)
            wav = mtl_model.generate(
                text=text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            pbar.update_absolute(90)

            audio_data = {
                "waveform": wav.unsqueeze(0),
                "sample_rate": mtl_model.sr
            }
            message += f"\nSpeech generated successfully (Multilingual)"
            return (audio_data, message)

        except RuntimeError as e:
            error_str = str(e)
            fallback_to_cpu = False
            if "CUDA" in error_str and device == "cuda":
                message += "\nCUDA error detected. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                message += "\nMPS error detected. Falling back to CPU..."
                fallback_to_cpu = True

            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model
                clear_cached_model("multilingual")

                message += f"\nLoading Multilingual TTS model on CPU..."
                pbar.update_absolute(10)
                mtl_model = load_multilingual_model(device=device)
                pbar.update_absolute(50)

                wav = mtl_model.generate(
                    text=text,
                    language_id=language_id,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
                pbar.update_absolute(90)

                audio_data = {
                    "waveform": wav.unsqueeze(0),
                    "sample_rate": mtl_model.sr
                }
                message += f"\nSpeech generated successfully after fallback (Multilingual)"
                return (audio_data, message)
            else:
                message += f"\nError during Multilingual TTS: {str(e)}"
                return (audio_data, message)
        except Exception as e:
            message += f"\nAn unexpected error occurred during Multilingual TTS: {str(e)}"
            return (audio_data, message)
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            # If keep_model_loaded is False, clear the cache
            if not keep_model_loaded:
                clear_cached_model("multilingual")

        pbar.update_absolute(100)
        return (audio_data, message)


# Voice Conversion node
class FL_ChatterboxVCNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Voice Conversion functionality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio": ("AUDIO",),
                "target_voice": ("AUDIO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox"
    
    def convert_voice(self, input_audio, target_voice, seed, use_cpu=False, keep_model_loaded=False):
        """
        Convert the voice in an audio file to match a target voice.

        Args:
            input_audio: AUDIO object containing the audio to convert.
            target_voice: AUDIO object containing the target voice.
            seed: Random seed for reproducible generation.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after conversion.

        Returns:
            Tuple of (audio, message)
        """
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
        # Determine device to use
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif torch.backends.mps.is_available() and device == "mps":
             message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
             message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference" # Should be CPU if no GPU found

        # Create temporary files for the audio inputs
        import tempfile
        temp_files = []

        # Create a temporary file for the input audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            input_audio_path = temp_input.name
            temp_files.append(input_audio_path)

        # Save the input audio to the temporary file
        input_waveform = input_audio['waveform'].squeeze(0)
        save_audio_wav(input_audio_path, input_waveform, input_audio['sample_rate'])

        # Create a temporary file for the target voice
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_target:
            target_voice_path = temp_target.name
            temp_files.append(target_voice_path)

        # Save the target voice to the temporary file
        target_waveform = target_voice['waveform'].squeeze(0)
        save_audio_wav(target_voice_path, target_waveform, target_voice['sample_rate'])

        vc_model = None
        pbar = ProgressBar(100) # Simple progress bar for overall process
        try:
            # Load the VC model or reuse if cached
            vc_model = get_cached_model("vc", device)
            if vc_model is not None:
                message += f"\nReusing loaded VC model on {device}..."
            else:
                # Clear any cached model on different device
                clear_cached_model("vc")

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started
                vc_model = load_vc_model(device=device)
                pbar.update_absolute(50) # Indicate model loading finished

                if keep_model_loaded:
                    cache_model("vc", device, vc_model)
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."

            # Convert voice
            message += f"\nConverting voice to match target voice"

            pbar.update_absolute(60) # Indicate conversion started
            converted_wav = vc_model.generate(
                audio=input_audio_path,
                target_voice_path=target_voice_path,
            )
            pbar.update_absolute(90) # Indicate conversion finished
            
        except RuntimeError as e:
            # Check for CUDA or MPS errors and attempt fallback to CPU
            error_str = str(e)
            fallback_to_cpu = False
            if "CUDA" in error_str and device == "cuda":
                message += "\nCUDA error detected during VC. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                 message += "\nMPS error detected during VC. Falling back to CPU..."
                 fallback_to_cpu = True

            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model
                clear_cached_model("vc")

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started (fallback)
                vc_model = load_vc_model(device=device)
                pbar.update_absolute(50) # Indicate model loading finished (fallback)
                # Note: keep_model_loaded logic is applied after successful generation
                # to avoid keeping a failed model loaded.

                converted_wav = vc_model.generate(
                    audio=input_audio_path,
                    target_voice_path=target_voice_path,
                )
                pbar.update_absolute(90) # Indicate conversion finished (fallback)
            else:
                # Re-raise if it's not a CUDA/MPS error or we're already on CPU
                message += f"\nError during VC: {str(e)}"
                # Return the original audio
                message += f"\nError: {str(e)}"
                pbar.update_absolute(100) # Ensure progress bar completes on error
                return (input_audio, message)
        except Exception as e:
             message += f"\nAn unexpected error occurred during VC: {str(e)}"
             empty_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000}
             for temp_file in temp_files:
                 if os.path.exists(temp_file):
                     os.unlink(temp_file)
             pbar.update_absolute(100) # Ensure progress bar completes on error
             return (empty_audio, message)
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            # If keep_model_loaded is False, clear the cache
            if not keep_model_loaded:
                clear_cached_model("vc")

        # Create audio data structure for the output
        audio_data = {
            "waveform": converted_wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": vc_model.sr if vc_model else 16000 # Use default sample rate if model loading failed
        }

        message += f"\nVoice converted successfully"
        pbar.update_absolute(100) # Ensure progress bar completes on success

        return (audio_data, message)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_ChatterboxTTS": FL_ChatterboxTTSNode,
    "FL_ChatterboxTurboTTS": FL_ChatterboxTurboTTSNode,
    "FL_ChatterboxMultilingualTTS": FL_ChatterboxMultilingualTTSNode,
    "FL_ChatterboxVC": FL_ChatterboxVCNode,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ChatterboxTTS": "FL Chatterbox TTS",
    "FL_ChatterboxTurboTTS": "FL Chatterbox Turbo TTS",
    "FL_ChatterboxMultilingualTTS": "FL Chatterbox Multilingual TTS",
    "FL_ChatterboxVC": "FL Chatterbox VC",
}
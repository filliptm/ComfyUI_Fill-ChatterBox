import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional

# Import directly from the chatterbox package
from .local_chatterbox.chatterbox.tts import ChatterboxTTS
from .local_chatterbox.chatterbox.vc import ChatterboxVC

from comfy.utils import ProgressBar
import folder_paths

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


def load_chatterbox_vc_from_local(ckpt_dir, device):
    """Custom loader for ChatterboxVC that handles both .pt and .safetensors formats"""
    from safetensors.torch import load_file
    
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

    # Import S3Gen from the same place ChatterboxVC imports it
    from .local_chatterbox.chatterbox.s3gen import S3Gen
    s3gen = S3Gen()
    
    # Try to load s3gen from either .pt or .safetensors
    s3gen_pt_path = ckpt_dir / "s3gen.pt"
    s3gen_safetensors_path = ckpt_dir / "s3gen.safetensors"
    
    if s3gen_pt_path.exists():
        s3gen.load_state_dict(torch.load(s3gen_pt_path, map_location=map_location))
    elif s3gen_safetensors_path.exists():
        s3gen.load_state_dict(load_file(s3gen_safetensors_path))
    else:
        raise FileNotFoundError("Neither s3gen.pt nor s3gen.safetensors found")
    
    s3gen.to(device).eval()

    return ChatterboxVC(s3gen, device, ref_dict=ref_dict)


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
    _tts_model = None
    _tts_device = None
    
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
                "use_custom_model": ("BOOLEAN", {"default": False}),
                "audio_prompt": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"
    
    def generate_speech(self, text, exaggeration, cfg_weight, temperature, seed, use_custom_model=False, audio_prompt=None, use_cpu=False, keep_model_loaded=False):
        """
        Generate speech from text.
        
        Args:
            text: The text to convert to speech.
            exaggeration: Controls emotion intensity (0.25-2.0).
            cfg_weight: Controls pace/classifier-free guidance (0.2-1.0).
            temperature: Controls randomness in generation (0.05-5.0).
            seed: Random seed for reproducible generation.
            use_custom_model: If True, uses custom model from ComfyUI/models/chatterbox/.
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
                torchaudio.save(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
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
            # Load the TTS model or reuse if loaded and device matches
            if FL_ChatterboxTTSNode._tts_model is not None and FL_ChatterboxTTSNode._tts_device == device:
                tts_model = FL_ChatterboxTTSNode._tts_model
                message += f"\nReusing loaded TTS model on {device}..."
            else:
                if FL_ChatterboxTTSNode._tts_model is not None:
                    message += f"\nUnloading previous TTS model (device mismatch or keep_model_loaded is False)..."
                    FL_ChatterboxTTSNode._tts_model = None
                    FL_ChatterboxTTSNode._tts_device = None
                    if torch.cuda.is_available():
                         torch.cuda.empty_cache() # Clear CUDA cache if possible
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache() # Clear MPS cache if possible


                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10)

                # Check for custom model path
                if use_custom_model:
                    model_path = os.path.join(folder_paths.models_dir, "chatterbox")
                    print(f"[ChatterboxTTS] Looking for custom model at: {model_path}")
                    print(f"[ChatterboxTTS] ComfyUI models directory: {folder_paths.models_dir}")
                    message += f"\nLooking for custom model at: {model_path}"
                    message += f"\nComfyUI models directory: {folder_paths.models_dir}"
                    if os.path.isdir(model_path):
                        print(f"[ChatterboxTTS] Custom model directory found: {model_path}")
                        message += f"\nCustom model directory found: {model_path}"
                        # List all files in the directory
                        try:
                            files_in_dir = os.listdir(model_path)
                            print(f"[ChatterboxTTS] Files in custom model directory: {files_in_dir}")
                            message += f"\nFiles in custom model directory: {files_in_dir}"
                        except Exception as e:
                            print(f"[ChatterboxTTS] Error listing files in directory: {str(e)}")
                            message += f"\nError listing files in directory: {str(e)}"
                        
                        # Check for required files
                        required_files = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
                        missing_files = []
                        for file in required_files:
                            if not os.path.exists(os.path.join(model_path, file)):
                                missing_files.append(file)
                        
                        if missing_files:
                            print(f"[ChatterboxTTS] ERROR: Missing required files: {missing_files}")
                            print(f"[ChatterboxTTS] Falling back to default model.")
                            message += f"\nError: Missing required files in custom model directory: {missing_files}"
                            message += f"\nFalling back to default model."
                            tts_model = ChatterboxTTS.from_pretrained(device=device)
                        else:
                            try:
                                print(f"[ChatterboxTTS] Attempting to load custom TTS model...")
                                message += f"\nAttempting to load custom TTS model..."
                                tts_model = ChatterboxTTS.from_local(model_path, device=device)
                                print(f"[ChatterboxTTS] ✅ CUSTOM TTS MODEL LOADED SUCCESSFULLY!")
                                print(f"[ChatterboxTTS] Model loaded from: {model_path}")
                                message += f"\n✅ CUSTOM TTS MODEL LOADED SUCCESSFULLY!"
                                message += f"\nModel loaded from: {model_path}"
                            except Exception as e:
                                print(f"[ChatterboxTTS] ❌ ERROR loading custom TTS model: {str(e)}")
                                print(f"[ChatterboxTTS] Falling back to default Hugging Face model.")
                                message += f"\n❌ ERROR loading custom TTS model: {str(e)}"
                                message += f"\nFalling back to default Hugging Face model."
                                tts_model = ChatterboxTTS.from_pretrained(device=device)
                    else:
                        print(f"[ChatterboxTTS] Warning: Custom model directory not found at {model_path}. Falling back to default.")
                        message += f"\nWarning: Custom model directory not found at {model_path}. Falling back to default."
                        tts_model = ChatterboxTTS.from_pretrained(device=device)
                else:
                    print(f"[ChatterboxTTS] Loading default model from Hugging Face.")
                    message += "\nLoading default model from Hugging Face."
                    tts_model = ChatterboxTTS.from_pretrained(device=device)

                pbar.update_absolute(50)

                if keep_model_loaded:
                    FL_ChatterboxTTSNode._tts_model = tts_model
                    FL_ChatterboxTTSNode._tts_device = device
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."

            # Generate speech
            message += f"\nGenerating speech for: {text[:50]}..." if len(text) > 50 else f"\nGenerating speech for: {text}"
            if audio_prompt_path:
                message += f"\nUsing audio prompt: {audio_prompt_path}"
            
            # Check if we're using custom model
            if use_custom_model:
                print(f"[ChatterboxTTS] 🔄 USING CUSTOM MODEL FOR GENERATION")
                message += f"\n🔄 USING CUSTOM MODEL FOR GENERATION"
            else:
                print(f"[ChatterboxTTS] 🔄 USING DEFAULT HUGGING FACE MODEL FOR GENERATION")
                message += f"\n🔄 USING DEFAULT HUGGING FACE MODEL FOR GENERATION"
            
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
                # Unload previous model if it exists
                if FL_ChatterboxTTSNode._tts_model is not None:
                    message += f"\nUnloading previous TTS model..."
                    FL_ChatterboxTTSNode._tts_model = None
                    FL_ChatterboxTTSNode._tts_device = None
                    if torch.cuda.is_available():
                         torch.cuda.empty_cache() # Clear CUDA cache if possible
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache() # Clear MPS cache if possible


                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10)
                
                if use_custom_model:
                    model_path = os.path.join(folder_paths.models_dir, "chatterbox")
                    if os.path.isdir(model_path):
                        # Check for required files
                        required_files = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
                        missing_files = []
                        for file in required_files:
                            if not os.path.exists(os.path.join(model_path, file)):
                                missing_files.append(file)
                        
                        if missing_files:
                            tts_model = ChatterboxTTS.from_pretrained(device=device)
                        else:
                            try:
                                tts_model = ChatterboxTTS.from_local(model_path, device=device)
                            except Exception as e:
                                tts_model = ChatterboxTTS.from_pretrained(device=device)
                    else:
                        tts_model = ChatterboxTTS.from_pretrained(device=device)
                else:
                    tts_model = ChatterboxTTS.from_pretrained(device=device)

                pbar.update_absolute(50)
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
            # If keep_model_loaded is False, ensure model is not stored
            # This is done here to ensure model is only kept if generation was successful
            if not keep_model_loaded and FL_ChatterboxTTSNode._tts_model is not None:
                 message += "\nUnloading TTS model as keep_model_loaded is False."
                 FL_ChatterboxTTSNode._tts_model = None
                 FL_ChatterboxTTSNode._tts_device = None
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache() # Clear CUDA cache if possible
                 if torch.backends.mps.is_available():
                     torch.mps.empty_cache() # Clear MPS cache if possible

        pbar.update_absolute(100) # Ensure progress bar completes on success or error
        return (audio_data, message) # Fallback return, should ideally not be reached


        # If generation was successful and keep_model_loaded is True, store the model
        if keep_model_loaded and tts_model is not None:
             FL_ChatterboxTTSNode._tts_model = tts_model
             FL_ChatterboxTTSNode._tts_device = device
             message += "\nModel will be kept loaded in memory."
        elif not keep_model_loaded and FL_ChatterboxTTSNode._tts_model is not None:
             # This case handles successful generation when keep_model_loaded was True previously
             # but is now False. Ensure the model is unloaded.
             message += "\nUnloading TTS model as keep_model_loaded is now False."
             FL_ChatterboxTTSNode._tts_model = None
             FL_ChatterboxTTSNode._tts_device = None
             if torch.cuda.is_available():
                 torch.cuda.empty_cache() # Clear CUDA cache if possible
             if torch.backends.mps.is_available():
                 torch.mps.empty_cache() # Clear MPS cache if possible


        # Create audio data structure for the output
        audio_data = {
            "waveform": wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": tts_model.sr if tts_model else 16000 # Use default sample rate if model loading failed
        }
        
        message += f"\nSpeech generated successfully"
        pbar.update_absolute(100) # Ensure progress bar completes on success
        
        return (audio_data, message)

# Voice Conversion node
class FL_ChatterboxVCNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Voice Conversion functionality.
    """
    _vc_model = None
    _vc_device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio": ("AUDIO",),
                "target_voice": ("AUDIO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "use_custom_model": ("BOOLEAN", {"default": False}),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox"
    
    def convert_voice(self, input_audio, target_voice, seed, use_custom_model=False, use_cpu=False, keep_model_loaded=False):
        """
        Convert the voice in an audio file to match a target voice.
        
        Args:
            input_audio: AUDIO object containing the audio to convert.
            target_voice: AUDIO object containing the target voice.
            seed: Random seed for reproducible generation.
            use_custom_model: If True, uses custom model from ComfyUI/models/chatterbox/.
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
        torchaudio.save(input_audio_path, input_waveform, input_audio['sample_rate'])
        
        # Create a temporary file for the target voice
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_target:
            target_voice_path = temp_target.name
            temp_files.append(target_voice_path)
        
        # Save the target voice to the temporary file
        target_waveform = target_voice['waveform'].squeeze(0)
        torchaudio.save(target_voice_path, target_waveform, target_voice['sample_rate'])
        
        vc_model = None
        pbar = ProgressBar(100) # Simple progress bar for overall process
        try:
            # Load the VC model or reuse if loaded and device matches
            if FL_ChatterboxVCNode._vc_model is not None and FL_ChatterboxVCNode._vc_device == device:
                vc_model = FL_ChatterboxVCNode._vc_model
                message += f"\nReusing loaded VC model on {device}..."
            else:
                if FL_ChatterboxVCNode._vc_model is not None:
                    message += f"\nUnloading previous VC model (device mismatch or keep_model_loaded is False)..."
                    FL_ChatterboxVCNode._vc_model = None
                    FL_ChatterboxVCNode._vc_device = None
                    if torch.cuda.is_available():
                         torch.cuda.empty_cache() # Clear CUDA cache if possible
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache() # Clear MPS cache if possible

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10)

                if use_custom_model:
                    model_path = os.path.join(folder_paths.models_dir, "chatterbox")
                    print(f"[ChatterboxVC] Looking for custom model at: {model_path}")
                    print(f"[ChatterboxVC] ComfyUI models directory: {folder_paths.models_dir}")
                    if os.path.isdir(model_path):
                        print(f"[ChatterboxVC] Custom VC model directory found: {model_path}")
                        message += f"\nCustom VC model directory found: {model_path}"
                        # List all files in the directory
                        try:
                            files_in_dir = os.listdir(model_path)
                            print(f"[ChatterboxVC] Files in custom VC model directory: {files_in_dir}")
                            message += f"\nFiles in custom VC model directory: {files_in_dir}"
                        except Exception as e:
                            print(f"[ChatterboxVC] Error listing files in directory: {str(e)}")
                            message += f"\nError listing files in directory: {str(e)}"
                        
                        # Check for required files for VC (can be either .pt or .safetensors)
                        has_s3gen = os.path.exists(os.path.join(model_path, "s3gen.pt")) or os.path.exists(os.path.join(model_path, "s3gen.safetensors"))
                        missing_files = []
                        if not has_s3gen:
                            missing_files.append("s3gen.pt or s3gen.safetensors")
                        
                        if missing_files:
                            print(f"[ChatterboxVC] ERROR: Missing required files: {missing_files}")
                            print(f"[ChatterboxVC] Falling back to default model.")
                            message += f"\nError: Missing required files in custom model directory: {missing_files}"
                            message += f"\nFalling back to default model."
                            vc_model = ChatterboxVC.from_pretrained(device=device)
                        else:
                            try:
                                print(f"[ChatterboxVC] Attempting to load custom VC model...")
                                message += f"\nAttempting to load custom VC model..."
                                vc_model = load_chatterbox_vc_from_local(model_path, device=device)
                                print(f"[ChatterboxVC] ✅ CUSTOM VC MODEL LOADED SUCCESSFULLY!")
                                print(f"[ChatterboxVC] Model loaded from: {model_path}")
                                message += f"\n✅ CUSTOM VC MODEL LOADED SUCCESSFULLY!"
                                message += f"\nModel loaded from: {model_path}"
                            except Exception as e:
                                print(f"[ChatterboxVC] ❌ ERROR loading custom VC model: {str(e)}")
                                print(f"[ChatterboxVC] Falling back to default Hugging Face model.")
                                message += f"\n❌ ERROR loading custom VC model: {str(e)}"
                                message += f"\nFalling back to default Hugging Face model."
                                vc_model = ChatterboxVC.from_pretrained(device=device)
                    else:
                        print(f"[ChatterboxVC] Warning: Custom model directory not found at {model_path}. Falling back to default.")
                        message += f"\nWarning: Custom model directory not found at {model_path}. Falling back to default."
                        vc_model = ChatterboxVC.from_pretrained(device=device)
                else:
                    print(f"[ChatterboxVC] Loading default model from Hugging Face.")
                    message += "\nLoading default model from Hugging Face."
                    vc_model = ChatterboxVC.from_pretrained(device=device)
                
                pbar.update_absolute(50)

                if keep_model_loaded:
                    FL_ChatterboxVCNode._vc_model = vc_model
                    FL_ChatterboxVCNode._vc_device = device
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
                # Unload previous model if it exists
                if FL_ChatterboxVCNode._vc_model is not None:
                    message += f"\nUnloading previous VC model..."
                    FL_ChatterboxVCNode._vc_model = None
                    FL_ChatterboxVCNode._vc_device = None
                    if torch.cuda.is_available():
                         torch.cuda.empty_cache() # Clear CUDA cache if possible
                    if torch.backends.mps.is_available():
                         torch.mps.empty_cache() # Clear MPS cache if possible

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10)

                if use_custom_model:
                    model_path = os.path.join(folder_paths.models_dir, "chatterbox")
                    if os.path.isdir(model_path):
                        # Check for required files for VC (can be either .pt or .safetensors)
                        has_s3gen = os.path.exists(os.path.join(model_path, "s3gen.pt")) or os.path.exists(os.path.join(model_path, "s3gen.safetensors"))
                        missing_files = []
                        if not has_s3gen:
                            missing_files.append("s3gen.pt or s3gen.safetensors")
                        
                        if missing_files:
                            vc_model = ChatterboxVC.from_pretrained(device=device)
                        else:
                            try:
                                vc_model = load_chatterbox_vc_from_local(model_path, device=device)
                            except Exception as e:
                                vc_model = ChatterboxVC.from_pretrained(device=device)
                    else:
                        vc_model = ChatterboxVC.from_pretrained(device=device)
                else:
                    vc_model = ChatterboxVC.from_pretrained(device=device)

                pbar.update_absolute(50)
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
            # If keep_model_loaded is False, ensure model is not stored
            # This is done here to ensure model is only kept if generation was successful
            if not keep_model_loaded and FL_ChatterboxVCNode._vc_model is not None:
                 message += "\nUnloading VC model as keep_model_loaded is False."
                 FL_ChatterboxVCNode._vc_model = None
                 FL_ChatterboxVCNode._vc_device = None
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache() # Clear CUDA cache if possible
                 if torch.backends.mps.is_available():
                     torch.mps.empty_cache() # Clear MPS cache if possible

        # If generation was successful and keep_model_loaded is True, store the model
        if keep_model_loaded and vc_model is not None:
             FL_ChatterboxVCNode._vc_model = vc_model
             FL_ChatterboxVCNode._vc_device = device
             message += "\nModel will be kept loaded in memory."
        elif not keep_model_loaded and FL_ChatterboxVCNode._vc_model is not None:
             # This case handles successful generation when keep_model_loaded was True previously
             # but is now False. Ensure the model is unloaded.
             message += "\nUnloading VC model as keep_model_loaded is now False."
             FL_ChatterboxVCNode._vc_model = None
             FL_ChatterboxVCNode._vc_device = None
             if torch.cuda.is_available():
                 torch.cuda.empty_cache() # Clear CUDA cache if possible
             if torch.backends.mps.is_available():
                 torch.mps.empty_cache() # Clear MPS cache if possible

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
    "FL_ChatterboxVC": FL_ChatterboxVCNode,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ChatterboxTTS": "FL Chatterbox TTS",
    "FL_ChatterboxVC": "FL Chatterbox VC",
}
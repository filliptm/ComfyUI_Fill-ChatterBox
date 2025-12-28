"""
Centralized path management for Chatterbox models.
All models download to: ComfyUI/models/chatterbox/
"""

from pathlib import Path


def get_chatterbox_models_dir() -> Path:
    """
    Get the centralized models directory for all Chatterbox models.

    Returns:
        Path to ComfyUI/models/chatterbox/
    """
    # Navigate: paths.py -> chatterbox -> local_chatterbox -> ComfyUI_Fill-ChatterBox -> custom_nodes -> ComfyUI
    current_dir = Path(__file__).parent
    comfyui_root = current_dir.parent.parent.parent.parent

    models_dir = comfyui_root / "models" / "chatterbox"

    # Verify we're in a valid ComfyUI structure
    if not (comfyui_root / "custom_nodes").exists():
        # Fallback: use a local directory if not in ComfyUI structure
        models_dir = current_dir.parent.parent / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_chatterbox_tts_dir() -> Path:
    """Get the directory for standard Chatterbox TTS models (English, 500M)."""
    path = get_chatterbox_models_dir() / "chatterbox"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_chatterbox_turbo_dir() -> Path:
    """Get the directory for Chatterbox Turbo TTS models (350M, faster)."""
    path = get_chatterbox_models_dir() / "chatterbox_turbo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_chatterbox_multilingual_dir() -> Path:
    """Get the directory for Chatterbox Multilingual TTS models (23 languages)."""
    path = get_chatterbox_models_dir() / "chatterbox_multilingual"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_to_local(repo_id: str, filenames: list, local_dir: Path) -> Path:
    """
    Download model files from HuggingFace to a local directory.

    Args:
        repo_id: HuggingFace repository ID (e.g., "ResembleAI/chatterbox")
        filenames: List of filenames to download
        local_dir: Local directory to download to

    Returns:
        Path to the local directory containing the downloaded files
    """
    from huggingface_hub import hf_hub_download
    import shutil

    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        local_path = local_dir / filename
        if not local_path.exists():
            print(f"[FL Chatterbox] Downloading {filename} to {local_dir}...")
            try:
                # Download to HF cache first, then copy to our location
                cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
                # Copy from cache to our directory
                shutil.copy2(cached_path, local_path)
            except Exception as e:
                print(f"[FL Chatterbox] Error downloading {filename}: {e}")
                raise
        else:
            print(f"[FL Chatterbox] Using cached {filename}")

    return local_dir

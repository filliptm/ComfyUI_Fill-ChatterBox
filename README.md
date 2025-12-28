# FL ChatterBox

High-quality text-to-speech nodes for ComfyUI powered by ResembleAI's Chatterbox models. Features voice cloning, multilingual synthesis, paralinguistic expressions, and voice conversion.

[![Chatterbox](https://img.shields.io/badge/Chatterbox-Original%20Repo-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/resemble-ai/chatterbox)
[![Patreon](https://img.shields.io/badge/Patreon-Support%20Me-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/Machinedelusions)

![Workflow Preview](assets/workflow_preview.png)

## Features

- **Zero-Shot Voice Cloning** - Clone any voice from a few seconds of reference audio
- **3 TTS Models** - Standard, Turbo (faster), and Multilingual variants
- **23 Languages** - Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish
- **Paralinguistic Tags** - Express emotions with tags like `[laugh]`, `[sigh]`, `[gasp]`, `[chuckle]` (Turbo model)
- **Voice Conversion** - Transform one voice to sound like another
- **Dialog Synthesis** - Multi-speaker conversations with up to 4 voices
- **Model Caching** - Keep models loaded between runs for faster iteration

## Nodes

| Node | Description |
|------|-------------|
| **FL Chatterbox TTS** | Standard high-quality text-to-speech with voice cloning |
| **FL Chatterbox Turbo TTS** | Faster GPT2-based TTS with paralinguistic tag support |
| **FL Chatterbox Multilingual TTS** | 23-language TTS with voice cloning |
| **FL Chatterbox VC** | Voice conversion - transform source audio to target voice |
| **FL Chatterbox Dialog TTS** | Multi-speaker dialog synthesis with up to 4 voices |

## Installation

### ComfyUI Manager
Search for "FL ChatterBox" and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI_Fill-ChatterBox.git
cd ComfyUI_Fill-ChatterBox
pip install -r requirements.txt
```

### Optional: Watermarking Support
```bash
pip install resemble-perth
```
**Note**: The `resemble-perth` package may have compatibility issues with Python 3.12+. Nodes will function without watermarking if import fails.

## Quick Start

1. Add **FL Chatterbox TTS** (or Turbo/Multilingual variant)
2. Enter your text in the text field
3. Optionally connect reference audio for voice cloning
4. Set `keep_model_loaded = True` for faster subsequent runs
5. Generate!

### Turbo Model with Expressions
```
Hello there! [laugh] Isn't this amazing? [sigh] I just love text to speech.
```
Supported tags: `[laugh]`, `[sigh]`, `[gasp]`, `[chuckle]`, `[cough]`, `[sniff]`, `[groan]`, `[shush]`, `[clear throat]`

## Models

| Model | Speed | Languages | Notes |
|-------|-------|-----------|-------|
| Standard | Normal | English | Highest quality |
| Turbo | Fast | English | Paralinguistic tags, GPT2-based |
| Multilingual | Normal | 23 languages | Cross-lingual voice cloning |

Models download automatically on first use to `ComfyUI/models/chatterbox/`.

## Parameters

### TTS Parameters
| Parameter | Range | Description |
|-----------|-------|-------------|
| `exaggeration` | 0.25-2.0 | Emotion intensity |
| `cfg_weight` | 0.2-1.0 | Pace/classifier-free guidance |
| `temperature` | 0.05-5.0 | Randomness in generation |
| `seed` | 0-4.29B | Reproducible generation |
| `keep_model_loaded` | bool | Cache model between runs |

### Turbo Parameters
| Parameter | Range | Description |
|-----------|-------|-------------|
| `temperature` | 0.05-2.0 | Randomness in generation |
| `top_k` | 1-5000 | Top-k sampling |
| `top_p` | 0.1-1.0 | Nucleus sampling threshold |
| `repetition_penalty` | 1.0-3.0 | Token repetition penalty |

## Limitations

- Maximum audio length: ~40 seconds per generation
- Reference audio: Minimum 5-6 seconds recommended
- Turbo paralinguistic tags: English only

## Requirements

- Python 3.10+
- 8GB RAM minimum (16GB+ recommended)
- NVIDIA GPU with 8GB+ VRAM recommended
- CPU and Mac MPS supported

## License

MIT License - See [Chatterbox repo](https://github.com/resemble-ai/chatterbox) for model licenses.

## Changelog

### 2025-12-28
- Added Turbo TTS node (faster, GPT2-based with paralinguistic tags)
- Added Multilingual TTS node (23 languages)
- Improved model caching using module-level globals
- Centralized model downloads to `ComfyUI/models/chatterbox/`

### 2025-07-24
- Added Dialog TTS node for multi-speaker conversations (up to 4 speakers)
- Extended all nodes with seed parameters for reproducible generation
- Isolated audio track outputs per speaker

### 2025-06-24
- Added seed parameter for reproducible generation
- Made Perth watermarking optional for Python 3.12+ compatibility

### 2025-05-31
- Added persistent model loading and loading bar
- Added Mac MPS support
- Native inference code (removed chatterbox-tts library dependency)

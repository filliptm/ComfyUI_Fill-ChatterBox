from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    start_text_token = 255
    stop_text_token = 0
    text_tokens_dict_size = 704
    max_text_tokens = 2048

    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 8194
    max_speech_tokens = 4096

    llama_config_name = "Llama_520M"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 150

    # For T3CondEnc
    encoder_type = "voice_encoder"
    speaker_embed_size = 256
    use_perceiver_resampler = True
    emotion_adv = True

    def __init__(self, text_tokens_dict_size=None):
        """Initialize T3Config with optional overrides."""
        if text_tokens_dict_size is not None:
            self.text_tokens_dict_size = text_tokens_dict_size

    @classmethod
    def multilingual(cls) -> 'T3Config':
        """Return config for multilingual model (23 languages)."""
        config = cls()
        config.text_tokens_dict_size = 2454
        config.llama_config_name = "Llama_520M"
        config.input_pos_emb = "learned"
        config.speech_cond_prompt_len = 150
        config.use_perceiver_resampler = True
        config.emotion_adv = True
        return config

    @classmethod
    def turbo(cls) -> 'T3Config':
        """Return config for turbo model (faster, GPT2-based)."""
        config = cls()
        config.text_tokens_dict_size = 50276
        config.llama_config_name = "GPT2_medium"
        config.speech_tokens_dict_size = 6563
        config.input_pos_emb = None
        config.speech_cond_prompt_len = 375
        config.use_perceiver_resampler = False
        config.emotion_adv = False
        return config

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

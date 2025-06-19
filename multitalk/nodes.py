import folder_paths
from comfy import model_management as mm
from comfy.utils import load_torch_file
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import torch

class MultiTalkModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MULTITALKMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision):
        from .multitalk import AudioProjModel

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)

        audio_proj_keys = [k for k in sd.keys() if "audio_proj" in k]
        audio_proj_sd = {k.replace("audio_proj.", ""): sd.pop(k) for k in audio_proj_keys}

        audio_window=5
        intermediate_dim=512
        output_dim=768
        context_tokens=32
        vae_scale=4
        norm_output_audio = True

        with init_empty_weights():
            multitalk_proj_model = AudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens,
                    norm_output_audio=norm_output_audio,
            )
        #fantasytalking_proj_model.load_state_dict(sd, strict=False)

        for name, param in multitalk_proj_model.named_parameters():
            set_module_tensor_to_device(multitalk_proj_model, name, device=offload_device, dtype=base_dtype, value=audio_proj_sd[name])

        multitalk = {
            "proj_model": multitalk_proj_model,
            "sd": sd,
        }

        return (multitalk,)
    

def loudness_norm(audio_array, sr=16000, lufs=-23):
    try:
        import pyloudnorm
    except:
        raise ImportError("pyloudnorm package is not installed")
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyloudnorm.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio
    
class MultiTalkWav2VecEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "wav2vec_model": ("WAV2VECMODEL",),
            "audio": ("AUDIO",),
            "normalize_loudness": ("BOOLEAN", {"default": True}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000, "step": 1}),
            "fps": ("FLOAT", {"default": 23.0, "min": 1.0, "max": 60.0, "step": 0.1}),
            "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Strength of the audio conditioning"}),
            "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "When not 1.0, an extra model pass without audio conditioning is done: slower inference but more motion is allowed"}),
            },
        }

    RETURN_TYPES = ("MULTITALK_EMBEDS", "AUDIO", )
    RETURN_NAMES = ("multitalk_embeds", "audio", )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, wav2vec_model, normalize_loudness, fps, num_frames, audio, audio_scale, audio_cfg_scale):
        import torchaudio
        import numpy as np
        from einops import rearrange

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = wav2vec_model["dtype"]
        wav2vec = wav2vec_model["model"]
        wav2vec_feature_extractor = wav2vec_model["feature_extractor"]

        sr = 16000

        audio_input = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if sample_rate != sr:
            audio_input = torchaudio.functional.resample(audio_input, sample_rate, sr)
        audio_input = audio_input[0][0]

        start_time = 0
        end_time = num_frames / fps

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        try:
            audio_segment = audio_input[start_sample:end_sample]
        except:
            audio_segment = audio_input

        audio_segment = audio_segment.numpy()

        if normalize_loudness:
            audio_segment = loudness_norm(audio_segment, sr=sr)

        audio_feature = np.squeeze(
            wav2vec_feature_extractor(audio_segment, sampling_rate=sr).input_values
        )

        audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio encoder
        audio_duration = len(audio_segment) / sr
        video_length = audio_duration * fps
        print("Audio duration:", audio_duration, "Video length:", video_length)
        embeddings = wav2vec(audio_feature.to(dtype), seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            print("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        multitalk_embeds = {
            "audio_features": audio_emb,
            "audio_scale": audio_scale,
            "audio_cfg_scale": audio_cfg_scale
        }

        audio_output = {
            "waveform": audio_feature.unsqueeze(0).cpu(),
            "sample_rate": sr
        }
    
        return (multitalk_embeds, audio_output)
    
NODE_CLASS_MAPPINGS = {
    "MultiTalkModelLoader": MultiTalkModelLoader,
    "MultiTalkWav2VecEmbeds": MultiTalkWav2VecEmbeds,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiTalkModelLoader": "MultiTalk Model Loader",
    "MultiTalkWav2VecEmbeds": "MultiTalk Wav2Vec Embeds",
}
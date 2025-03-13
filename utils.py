import importlib.metadata
import torch
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from accelerate.utils import set_module_tensor_to_device
def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.31.0'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")

def print_memory(device):
    memory = torch.cuda.memory_allocated(device) / 1024**3
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    log.info(f"Allocated memory: {memory=:.3f} GB")
    log.info(f"Max allocated memory: {max_memory=:.3f} GB")
    log.info(f"Max reserved memory: {max_reserved=:.3f} GB")
    #memory_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    #log.info(f"Memory Summary:\n{memory_summary}")

def get_module_memory_mb(module):
    memory = 0
    for param in module.parameters():
        if param.data is not None:
            memory += param.nelement() * param.element_size()
    return memory / (1024 * 1024)  # Convert to MB

def apply_lora(model, device_to, transformer_load_device, params_to_keep=None, dtype=None, base_dtype=None, state_dict=None, low_mem_load=False):
        to_load = []
        for n, m in model.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                to_load.append((n, m, params))

        to_load.sort(reverse=True)
        for x in tqdm(to_load, desc="Loading model and applying LoRA weights:", leave=True):
            name = x[0]
            m = x[1]
            params = x[2]
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights == True:
                    continue
            for param in params:
                if low_mem_load:
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if name.startswith("diffusion_model."):
                        name_no_prefix = name[len("diffusion_model."):]
                    key = "{}.{}".format(name_no_prefix, param)
                    set_module_tensor_to_device(model.model.diffusion_model, key, device=transformer_load_device, dtype=dtype_to_use, value=state_dict[key])
                model.patch_weight_to_device("{}.{}".format(name, param), device_to=device_to)
                if low_mem_load:
                    set_module_tensor_to_device(model.model.diffusion_model, key, device=transformer_load_device, dtype=dtype_to_use, value=model.model.diffusion_model.state_dict()[key])
    
            m.comfy_patched_weights = True
      
        model.current_weight_patches_uuid = model.patches_uuid
        if low_mem_load:
            for name, param in model.model.diffusion_model.named_parameters():
                if param.device != transformer_load_device:
                    #print("param.device", param.device)
                    set_module_tensor_to_device(model.model.diffusion_model, name, device=transformer_load_device, dtype=dtype_to_use, value=state_dict[name])
        return model
        

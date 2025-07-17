import torch
try:
    from sparse_sageattn import sparse_sageattn
except:
    sparse_sageattn = None
    raise ImportError("Package is not installed: https://github.com/jt-zhang/Sparse_SageAttention_API")
from einops import rearrange, repeat

def sparge_mask_convert(mask: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    assert block_size in [128, 64], "Radial Attention only supports block size of 128 or 64"
    assert mask.shape[0] == mask.shape[1], "Input mask must be square."

    if block_size == 128:
        new_mask = torch.repeat_interleave(mask, 2, dim=1)
        
    elif block_size == 64:
        num_row, num_col = mask.shape
        reshaped_mask = mask.view(num_row // 2, 2, num_col)
        new_mask = torch.max(reshaped_mask, dim=1).values

    return new_mask

from comfy import model_management as mm
device = mm.get_torch_device()

def get_indptr_from_mask(mask):
    # query shows the device of the indptr
    # indptr (torch.Tensor) - the block index pointer of the block-sparse matrix on row dimension,
    # shape `(MB + 1,)`, where `MB` is the number of blocks in the row dimension.
    # The first element is always 0, and the last element is the number of blocks in the row dimension.
    # The rest of the elements are the number of blocks in each row.
    # the mask is already a block sparse mask
    indptr = torch.zeros(mask.shape[0] + 1, device=device, dtype=torch.int32)
    indptr[0] = 0
    row_counts = mask.sum(dim=1).flatten()  # Ensure 1D output [num_blocks_row]
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return indptr

def get_indices_from_mask(mask):
    # indices (torch.Tensor) - the block indices of the block-sparse matrix on column dimension,
    # shape `(nnz,),` where `nnz` is the number of non-zero blocks.
    # The elements in `indices` array should be less than `NB`: the number of blocks in the column dimension.
    nonzero_indices = torch.nonzero(mask)
    indices = nonzero_indices[:, 1].to(dtype=torch.int32, device=device)
    return indices

def shrinkMaskStrict(mask, block_size=128):
    seqlen = mask.shape[0]
    block_num = seqlen // block_size
    mask = mask[:block_num * block_size, :block_num * block_size].view(block_num, block_size, block_num, block_size)
    col_densities = mask.sum(dim = 1) / block_size
    # we want the minimum non-zero column density in the block
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1/3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
    block_mask = frac_high_density_cols > 0.6
    block_mask[0:0] = True
    block_mask[-1:-1] = True
    return block_mask

def pad_qkv(input_tensor, block_size=128):
    """
    Pad the input tensor to be a multiple of the block size.
    input shape: (seqlen, num_heads, hidden_dim)
    """
    seqlen, num_heads, hidden_dim = input_tensor.shape
    # Calculate the necessary padding
    padding_length = (block_size - (seqlen % block_size)) % block_size
    # Create a padded tensor with zeros
    padded_tensor = torch.zeros((seqlen + padding_length, num_heads, hidden_dim), device=input_tensor.device, dtype=input_tensor.dtype)
    # Copy the original tensor into the padded tensor
    padded_tensor[:seqlen, :, :] = input_tensor
    
    return padded_tensor

def get_diagonal_split_mask(i, j, token_per_frame, sparse_type):
    assert(sparse_type in ["radial"])
    dist = abs(i - j)
    group = dist.bit_length()
    threshold = 128 # hardcoded threshold for now, which is equal to block-size
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group
    if decay_length >= threshold:
        return torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
    
    split_factor = int(threshold / decay_length)
    modular = dist % split_factor
    if modular == 0:
        return torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
    else:
        return torch.zeros((token_per_frame, token_per_frame), device=device, dtype=torch.bool)

def get_window_width(i, j, token_per_frame, sparse_type, decay_factor=1, block_size=128):
    assert(sparse_type in ["radial"])
    dist = abs(i - j)
    if dist < 1:
        return token_per_frame
    if dist == 1:
        return token_per_frame // 2
    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * decay_factor
    threshold = block_size
    if decay_length >= threshold:
        return decay_length
    else:
        return threshold

def gen_log_mask_shrinked(device, s, video_token_num, num_frame, block_size=128, sparse_type="log", decay_factor=0.5):
    """
    A more memory friendly version, we generate the attention mask of each frame pair at a time,
    shrinks it, and stores it into the final result
    """
    final_log_mask = torch.zeros((s // block_size, s // block_size), device=device, dtype=torch.bool)
    token_per_frame = video_token_num // num_frame
    video_text_border = video_token_num // block_size

    col_indices = torch.arange(0, token_per_frame, device=device).view(1, -1)
    row_indices = torch.arange(0, token_per_frame, device=device).view(-1, 1)
    final_log_mask[video_text_border:] = True
    final_log_mask[:, video_text_border:] = True
    for i in range(num_frame):
        for j in range(num_frame):
            local_mask = torch.zeros((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
            if j == 0: # this is attention sink
                local_mask = torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
            else:
                window_width = get_window_width(i, j, token_per_frame, sparse_type, decay_factor=decay_factor, block_size=block_size)
                local_mask = torch.abs(col_indices - row_indices) <= window_width
                split_mask = get_diagonal_split_mask(i, j, token_per_frame, sparse_type)
                local_mask = torch.logical_and(local_mask, split_mask)

            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size
            # get the padded size
            all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
            all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
            padded_local_mask = torch.zeros((all_length_row, all_length_col), device=device, dtype=torch.bool)
            padded_local_mask[remainder_row:remainder_row + token_per_frame, remainder_col:remainder_col + token_per_frame] = local_mask
            # shrink the mask
            block_mask = shrinkMaskStrict(padded_local_mask, block_size=block_size)
            # set the block mask to the final log mask
            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]
            final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(
                final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end], block_mask)
    #print(f"mask sparsity: {1 - final_log_mask.sum() / final_log_mask.numel()}")
    return final_log_mask

class MaskMap:

    def __init__(self, video_token_num=25440, num_frame=16):
        self.video_token_num = video_token_num
        self.num_frame = num_frame
        self.log_mask = None

    def queryLogMask(self, seq_len, sparse_type, block_size=128, decay_factor=0.5):
        log_mask = torch.ones((seq_len // block_size, seq_len // block_size), device=device, dtype=torch.bool)
        if self.log_mask is None:
            self.log_mask = gen_log_mask_shrinked(device, seq_len, self.video_token_num, self.num_frame, sparse_type=sparse_type, decay_factor=decay_factor, block_size=block_size)
        block_bound = self.video_token_num // block_size
        log_mask[:block_bound, :block_bound] = self.log_mask[:block_bound, :block_bound]
        return log_mask

def SpargeSageAttnBackend(query, key, value, mask_map=None, video_mask=None, block_size=128):
    if video_mask.all():
        # dense case
        output_video = sparse_sageattn(
            query[:,:mask_map.video_token_num],
            key[:,:key.shape[1], :, :],
            value[:,:key.shape[1], :, :],
            mask_id=None,
            is_causal=False,
            tensor_layout="NHD",
        )[0]

        return output_video.unsqueeze(0).contiguous()
    
    converted_mask = repeat(sparge_mask_convert(mask=video_mask, block_size=block_size), "s t -> b h s t", b=1, h=query.shape[-2])
    
    converted_mask = converted_mask.to(torch.int8)
    
    output = sparse_sageattn(
        query.transpose(1, 2)[:, :, :mask_map.video_token_num, :],
        key.transpose(1, 2)[:, :, :mask_map.video_token_num, :],
        value.transpose(1, 2)[:, :, :mask_map.video_token_num, :],
        mask_id=converted_mask,
        is_causal=False,
        tensor_layout="HND",
    )
    
    return output.transpose(1, 2).contiguous()

def RadialAttention(query, key, value, mask_map=None, sparsity_type="radial", block_size=128, decay_factor=1, use_sage_attention=False):
    if sparsity_type == "dense":
        video_mask = torch.ones((mask_map.video_token_num // block_size, mask_map.video_token_num // block_size), device=device, dtype=torch.bool)
    else:
        video_mask = mask_map.queryLogMask(query.shape[0] * query.shape[1], sparsity_type, block_size=block_size, decay_factor=decay_factor) if mask_map else None
    
    return SpargeSageAttnBackend(query, key, value, mask_map, video_mask, block_size=block_size)
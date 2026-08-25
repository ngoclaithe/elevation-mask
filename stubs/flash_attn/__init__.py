"""CPU stub so Florence-2 remote code can import without NVIDIA flash-attn."""


def flash_attn_func(*_args, **_kwargs):
    raise RuntimeError("flash_attn is stubbed; use eager attention")


flash_attn_varlen_func = flash_attn_func
flash_attn_qkvpacked_func = flash_attn_func
flash_attn_varlen_qkvpacked_func = flash_attn_func

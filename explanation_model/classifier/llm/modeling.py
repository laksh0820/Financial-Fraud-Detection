"""Model / tokenizer loading helpers for the LLM classifier."""
import logging


def pick_dtype():
    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return (torch.bfloat16 if use_bf16 else torch.float16), use_bf16


def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(base_model, load_in_4bit, dtype):
    from transformers import AutoModelForCausalLM
    kwargs = dict(device_map="auto", attn_implementation="sdpa")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = dtype
    logging.info(f"Loading base model {base_model} ...")
    return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

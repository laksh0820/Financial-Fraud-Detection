"""LLMPatternClassifier: (optionally LoRA-fine-tuned) causal LLM that names the laundering pattern."""
import logging
from typing import List, Optional

from explanation_model.classifier.base import ClassificationResult, PatternClassifier
from explanation_model.classifier.llm.modeling import load_base_model, load_tokenizer, pick_dtype
from explanation_model.classifier.llm.prompts import build_prompt, parse_pattern, render_prompt
from explanation_model.subgraph import Subgraph


class LLMPatternClassifier(PatternClassifier):
    """The model is loaded lazily on first use, so building the object is free (detect.py only pays
    for the LLM if a transaction is actually predicted as fraud).

    adapter_path=None -> plain base model (zero-shot baseline).
    """
    name = "llm"

    def __init__(self, base_model="Qwen/Qwen3-14B", adapter_path: Optional[str] = None,
                 load_in_4bit=False, max_new_tokens=24, batch_size=4):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    # ---- CLI integration ----------------------------------------------------------------------

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--llm_model", default="Qwen/Qwen3-14B", help="Base LLM (HF hub id or local path)")
        parser.add_argument("--adapter_path", default=None,
                            help="LoRA adapter dir produced by `python -m explanation_model.classifier.llm` "
                                 "(omit for the zero-shot base model)")
        parser.add_argument("--load_in_4bit", action="store_true", help="Load the LLM in 4-bit (bitsandbytes)")
        parser.add_argument("--llm_max_new_tokens", type=int, default=24)

    @classmethod
    def from_args(cls, args):
        return cls(base_model=args.llm_model, adapter_path=args.adapter_path,
                   load_in_4bit=args.load_in_4bit, max_new_tokens=args.llm_max_new_tokens)

    # ---- model --------------------------------------------------------------------------------

    def _load(self):
        if self.model is not None:
            return
        dtype, _ = pick_dtype()
        self.tokenizer = load_tokenizer(self.base_model)
        self.tokenizer.padding_side = "left"
        model = load_base_model(self.base_model, self.load_in_4bit, dtype)
        if self.adapter_path is None:
            logging.info("No adapter given: using the base model WITHOUT fine-tuning (zero-shot).")
        else:
            from peft import PeftModel
            logging.info(f"Loading LoRA adapter from {self.adapter_path}")
            model = PeftModel.from_pretrained(model, self.adapter_path)
            if not self.load_in_4bit:
                model = model.merge_and_unload()
        self.model = model.eval()

    def generate(self, prompts: List[str], log_progress=False) -> List[str]:
        """Greedy generation for already-built user prompts. Returns the raw (unstripped) outputs."""
        import torch
        self._load()
        outputs = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            texts = [render_prompt(self.tokenizer, p) for p in batch]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None, pad_token_id=self.tokenizer.pad_token_id,
                )
            outputs.extend(self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
            if log_progress:
                logging.info(f"[{min(i + self.batch_size, len(prompts))}/{len(prompts)}] done")
        return outputs

    # ---- PatternClassifier API ----------------------------------------------------------------

    def classify_batch(self, subgraphs: List[Subgraph]) -> List[ClassificationResult]:
        raws = self.generate([build_prompt(sg.serialize()) for sg in subgraphs])
        return [ClassificationResult(pattern=parse_pattern(r), raw=r.strip()) for r in raws]

    def classify(self, subgraph: Subgraph) -> ClassificationResult:
        return self.classify_batch([subgraph])[0]

"""
RAG Generation Module — Qwen2.5-0.5B-Instruct (or any Qwen-Instruct model).

Usage
-----
    from generation.generator import RAGGenerator

    gen = RAGGenerator("Qwen/Qwen2.5-0.5B-Instruct")

    response = gen.generate(
        query="What is tempura?",
        context_chunks=[{"text": "Tempura is a Japanese dish..."}],
        system_prompt="You are a helpful culinary assistant...",
    )

    # Fill responses into a full pipeline output payload
    output = gen.fill_responses(pipeline_output, system_prompt=SYSTEM_PROMPT)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Prompt template: context passages + question → user turn
_USER_TEMPLATE = """\
Context:
{context}

Question: {query}"""


def _resolve_model(
    model_ref: str,
    checkpoints_dir: Path = Path("model_checkpoints"),
) -> str:
    """Local-first model resolution (same pattern as the rest of the pipeline)."""
    p = Path(model_ref)
    if p.exists() and p.is_dir():
        return model_ref
    for name in (p.name, str(p).replace("/", "__")):
        candidate = checkpoints_dir / name
        if candidate.exists() and candidate.is_dir():
            print(f"[Generator] Local checkpoint: {candidate.resolve()}")
            return str(candidate)
    print(f"[Generator] Using HuggingFace: {model_ref}")
    return model_ref


def _format_context(chunks: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """
    Render retrieved chunks as a numbered passage list.
    Truncates to *max_chars* total to stay within the model's context window.
    """
    lines, total = [], 0
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        line = f"[{i}] {text}"
        if total + len(line) > max_chars:
            remaining = max_chars - total
            if remaining > 40:
                lines.append(line[:remaining] + "…")
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


class RAGGenerator:
    """
    Wraps a Qwen-Instruct model for retrieval-augmented generation.

    Parameters
    ----------
    model_ref:       HuggingFace ID or local path (auto-resolved).
    max_new_tokens:  Default token budget for generation.
    context_max_chars: Character cap for the context block in the prompt.
    device:          "cuda", "cpu", or None (auto-detect).
    """

    def __init__(
        self,
        model_ref: str = DEFAULT_MODEL,
        max_new_tokens: int = 256,
        context_max_chars: int = 2000,
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.context_max_chars = context_max_chars

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        resolved = _resolve_model(model_ref)
        print(f"[Generator] Loading model on {self.device}: {resolved}")

        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.float32,
            device_map=self.device,
        )
        self.model.eval()
        print(f"[Generator] Ready.")

    #  single query 

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response for one query given retrieved context chunks.

        Args:
            query:          User query string.
            context_chunks: List of dicts with at least a ``"text"`` key
                            (direct output of ``pipeline.retrieve()``).
            system_prompt:  System instruction — editable in the notebook.
            max_new_tokens: Override default token budget.

        Returns:
            Generated response string (decoded, special tokens stripped).
        """
        import torch

        context_str = _format_context(context_chunks, self.context_max_chars)
        user_content = _USER_TEMPLATE.format(context=context_str, query=query)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,           # greedy — deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # decode only the newly generated tokens
        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    #  batch payload 

    def fill_responses(
        self,
        pipeline_output: Dict[str, Any],
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fill in the ``response`` field for every item in a pipeline output payload.

        Input/output schema matches ``output_payload_sample.json``:
          {"results": [{"query_id": ..., "query": ..., "response": "",
                         "retrieved_context": [...]}]}

        Returns an updated copy of *pipeline_output* with responses filled.
        """
        import copy
        output = copy.deepcopy(pipeline_output)
        results = output.get("results", [])
        for i, item in enumerate(results):
            print(f"  Generating [{i+1}/{len(results)}] {item['query'][:60]}...")
            item["response"] = self.generate(
                query=item["query"],
                context_chunks=item["retrieved_context"],
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
            )
        return output

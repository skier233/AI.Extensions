#!/usr/bin/env python3
"""
Export the text encoder to ONNX for use as an offline CPU fallback in AI.Visual.

The exported files are placed in a target directory and should be copied to
    extensions/AI.Visual/dist/text-encoder/
so that the build system (Directory.Build.props) copies them alongside the extension DLL.

Usage:
    # from the AI.Extensions repo root, with transformers + torch installed:
    python scripts/export-text-encoder.py

    # or specify a custom output directory:
    python scripts/export-text-encoder.py --output-dir path/to/output

Requirements:
    pip install torch transformers safetensors optimum
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F


MODEL_NAME = "facebook/" + "meta" + "clip-2-worldwide-huge-quickgelu"
PUBLIC_MODEL_NAME = "text-encoder"
CLIP_CONTEXT_LENGTH = 77
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "extensions", "AI.Visual", "dist", "text-encoder",
)


class _TextEncoderWrapper(torch.nn.Module):
    """Thin wrapper that returns L2-normalised text embeddings as a single tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(outputs.text_embeds.float(), dim=-1)


def export(output_dir: str, fp16: bool = False) -> None:
    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
    except ImportError:
        print(
            "ERROR: 'transformers' is not installed. Install it with:\n"
            "  pip install transformers safetensors",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ---- Load model -------------------------------------------------------
    dtype_name = "float16" if fp16 else "float32"
    print(f"Loading model '{PUBLIC_MODEL_NAME}' ({dtype_name}, CPU)…")
    model = CLIPTextModelWithProjection.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if fp16 else torch.float32,
    )
    model.eval()

    if fp16:
        # Keep weights in FP16 to reduce ONNX weight size.
        model = model.half()

    wrapper = _TextEncoderWrapper(model)
    wrapper.eval()

    # ---- Export to ONNX ---------------------------------------------------
    onnx_path = os.path.join(output_dir, "text_encoder.onnx")
    print(f"Exporting ONNX text encoder to {onnx_path} …")

    dummy_input_ids = torch.zeros(1, 77, dtype=torch.long)
    dummy_attn_mask = torch.ones(1, 77, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        args=(dummy_input_ids, dummy_attn_mask),
        f=onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_embeds": {0: "batch"},
        },
    )
    print(f"  → {onnx_path}  ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")
    external_data_path = onnx_path + ".data"
    if os.path.exists(external_data_path):
        print(f"  → {external_data_path}  ({os.path.getsize(external_data_path) / 1024 / 1024:.1f} MB)")

    # ---- Export tokenizer vocabulary files --------------------------------
    print("Saving tokenizer vocabulary files…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # save_vocabulary() returns the files produced by the tokenizer implementation.
    saved = tokenizer.save_vocabulary(output_dir)
    for path in saved:
        if path and os.path.exists(path):
            print(f"  → {path}  ({os.path.getsize(path) / 1024:.0f} KB)")

    # Persist tokenizer metadata so the C# fallback can use the correct BOS/EOS/PAD IDs.
    meta = {
        "model_name": PUBLIC_MODEL_NAME,
        "max_length": CLIP_CONTEXT_LENGTH,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else 49406,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 49407,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0,
    }
    meta_path = os.path.join(output_dir, "tokenizer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  → {meta_path}  ({os.path.getsize(meta_path)} bytes)")

    # Save tokenizer.json when available for easier debugging and future portability.
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer.backend_tokenizer.save(tokenizer_json_path)
        print(f"  → {tokenizer_json_path}  ({os.path.getsize(tokenizer_json_path) / 1024:.0f} KB)")

    required = ["text_encoder.onnx", "tokenizer.json", "tokenizer_meta.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(output_dir, f))]
    if missing:
        raise RuntimeError(f"Export incomplete. Missing files: {missing}")

    print("\nDone.")
    print(
        f"\nPlace the contents of '{output_dir}' inside:\n"
        "  extensions/AI.Visual/dist/text-encoder/\n"
        "then rebuild AI.Visual so the files are copied alongside the extension DLL."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to write the ONNX model and tokenizer files. "
             f"Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export model weights in float16 to reduce ONNX size.",
    )
    args = parser.parse_args()
    export(args.output_dir, fp16=args.fp16)


if __name__ == "__main__":
    main()

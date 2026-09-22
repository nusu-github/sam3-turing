"""Compare complete 24-layer C++ VE text encoder against original checkpoints."""
import argparse
import json
from pathlib import Path

import torch
from sam3.model_builder import _create_text_encoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, help="sam3=/path/model.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.manual_seed(881)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    bpe = Path(__file__).resolve().parents[2] / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    for spec in args.checkpoint:
        label, filename = spec.split("=", 1)
        state = torch.load(filename, map_location="cpu", weights_only=True, mmap=True)
        state = state.get("model", state)
        prefix = "detector.backbone.language_backbone."
        reference = _create_text_encoder(str(bpe)).eval()
        reference.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
        reference.to(args.device)
        prompts = ["truck", "wheel", "a yellow butterfly", "赤い車"]
        cases = [
            ("prompts", reference.tokenizer(prompts, context_length=32)),
            ("dynamic-length", torch.randint(0, 49408, (2, 7))),
            ("single-token", torch.tensor([[49406]])),
        ]
        for case, tokens in cases:
            tokens = tokens.to(args.device)
            inputs_embeds = reference.encoder.token_embedding(tokens)
            _, memory = reference.encoder(tokens)
            expected = (tokens.eq(0), reference.resizer(memory.transpose(0, 1)), inputs_embeds.transpose(0, 1))
            actual = torch.ops.sam3_native.text_encode(str(args.store), label, tokens)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
            diff = (actual[1] - expected[1]).abs()
            result = {"model": label, "case": case, "tokens": list(tokens.shape), "max_abs_error": diff.max().item(), "mean_abs_error": diff.mean().item()}
            print(json.dumps(result), flush=True)
            torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=2e-5)
            results.append(result)
        del reference, state
    report = {"device": args.device, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None, "torch": torch.__version__, "tf32": False, "dtype": "float32", "rtol": 2e-5, "atol": 2e-5, "cases": results, "scope": "Token IDs to VE outputs; native Unicode/BPE tokenization remains unimplemented."}
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

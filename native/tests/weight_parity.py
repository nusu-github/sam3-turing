"""Exercise Python exporter -> independent C++ reader across every format dtype."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from export_weights import DTYPES, export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    with tempfile.TemporaryDirectory(prefix="sam3-weight-parity-") as tmp:
        root = Path(tmp)
        first = {}
        for name in DTYPES:
            tensor = torch.arange(6).reshape(2, 3).to(getattr(torch, name))
            if tensor.is_complex():
                tensor += torch.tensor(2j)
            first[f"detector.test.{name}"] = tensor.T  # noncontiguous source
        first["detector.test.scalar"] = torch.tensor(3.5)
        first["detector.test.empty"] = torch.empty(0, 3)
        second = {"tracker.shared": first["detector.test.float32"], "tracker.unique": torch.tensor([-5., 1., 2.])}
        paths = {"sam3": root / "sam3.pt", "sam3.1": root / "sam3.1.pt"}
        torch.save(first, paths["sam3"])
        torch.save(second, paths["sam3.1"])
        inventory_path = root / "inventory.json"
        subprocess.run([sys.executable, str(TOOLS / "inventory_weights.py"), "--checkpoint", f"sam3={paths['sam3']}", "--checkpoint", f"sam3.1={paths['sam3.1']}", "--output", str(inventory_path)], check=True, capture_output=True, text=True)
        inventory = json.loads(inventory_path.read_text())
        output = root / "store"
        export(inventory, paths, output)
        manifest = json.loads((output / "manifest.json").read_text())
        shared_id = manifest["models"]["sam3"]["detector.test.float32"]
        assert manifest["models"]["sam3.1"]["tracker.shared"] == shared_id
        assert manifest["tensors"][shared_id]["shard"] == "shared.s3w"
        cases = 0
        devices = ["cpu", "cuda"] if args.cuda else ["cpu"]
        for label, state in [("sam3", first), ("sam3.1", second)]:
            for name, expected in state.items():
                for device in devices:
                    actual = torch.ops.sam3_native.read_weight(str(output), label + "/" + name, device)
                    assert actual.device.type == device
                    assert actual.shape == expected.shape and actual.dtype == expected.dtype
                    # Compare raw logical bytes, including BF16/complex/scalar/empty.
                    a = actual.cpu().contiguous().reshape(-1).view(torch.uint8)
                    b = expected.contiguous().reshape(-1).view(torch.uint8)
                    assert torch.equal(a, b)
                    cases += 1
        # Stale alias must fail even though its first representative is unchanged.
        second["tracker.shared"] = second["tracker.shared"] + 1
        torch.save(second, paths["sam3.1"])
        try:
            export(inventory, paths, root / "stale")
        except ValueError:
            assert not (root / "stale").exists()
            assert not list(root.glob("stale-*"))
        else:
            raise AssertionError("stale inventory accepted")
        report = {"exact_tensor_roundtrips": cases, "dtype_count": len(DTYPES), "cross_model_sharing": "passed", "stale_alias_rejection": "passed", "devices": devices}
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

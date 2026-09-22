"""Verify head selection and restoration of a caller's autocast state."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library",type=Path)
    parser.add_argument("store",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    image = torch.linspace(-1,1,3*1008*1008,device="cuda").reshape(1,3,1008,1008)
    full = torch.ops.sam3_native.vision_encode(str(args.store),"sam3.1",image,"fp16",[])
    with torch.autocast("cuda",dtype=torch.bfloat16):
        selected = torch.ops.sam3_native.vision_encode(str(args.store),"sam3.1",image,"fp16",["propagation_convs"])
        assert torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
        try:
            torch.ops.sam3_native.vision_encode(str(args.store),"sam3.1",image,"fp16",["unknown_head"])
        except RuntimeError as error:
            assert "unknown vision head" in str(error)
        else:
            raise AssertionError("unknown head accepted")
        assert torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
    expected_keys = {k for k in full if k == "trunk" or k.startswith(("propagation_convs.","position."))}
    assert set(selected) == expected_keys
    for key in selected:
        torch.testing.assert_close(selected[key],full[key],rtol=0,atol=0)
    report = {"model":"sam3.1","selected_head":"propagation_convs","matching_outputs":sorted(selected),"all_values_exact":True,"outer_bf16_autocast_restored":True,"autocast_restored_after_error":True}
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

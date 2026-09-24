"""Native visual trunk/neck comparison, with explicit mixed-precision policy."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2
from sam3.model_builder import _create_vit_backbone
from sam3.model.necks import Sam3DualViTDetNeck, Sam3TriViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine


def plain_mlp(module, x):
    return module.fc2(module.act(module.fc1(x)))


def flat_output(output, tri):
    names = ["convs", "interactive_convs", "propagation_convs"] if tri else ["convs", "sam2_convs"]
    result = {}
    for i, name in enumerate(names):
        for level, value in enumerate(output[2 * i]):
            result[f"{name}.{level}"] = getattr(value, "tensors", value)
    for level, pos in enumerate(output[1]):
        result[f"position.{level}"] = pos
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--image", type=Path, default=Path("assets/images/truck.jpg"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    parser.add_argument("--modes", nargs="+", default=["bf16_reference", "fp32", "fp16"])
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    transform = v2.Compose([v2.ToDtype(torch.uint8, scale=True), v2.Resize((1008,1008)), v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5]*3,[0.5]*3)])
    image = transform(v2.functional.to_image(Image.open(args.image).convert("RGB")).to(args.device))[None]
    results = []
    for spec in args.checkpoint:
        label, filename = spec.split("=",1)
        tri = label == "sam3.1"
        cls = Sam3TriViTDetNeck if tri else Sam3DualViTDetNeck
        options = {} if tri else {"add_sam2_neck": True}
        reference = cls(trunk=_create_vit_backbone(), position_encoding=PositionEmbeddingSine(256), d_model=256,
                        scale_factors=[4.,2.,1.] if tri else [4.,2.,1.,0.5], **options).eval()
        state = torch.load(filename,map_location="cpu",weights_only=True,mmap=True)
        state = state.get("model",state)
        prefix = "detector.backbone.vision_backbone."
        reference.load_state_dict({key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)},strict=True)
        reference.to(args.device)
        original_mlps = [block.mlp.forward for block in reference.trunk.blocks]
        trunk = []
        handle = reference.trunk.register_forward_hook(lambda module, inputs, output: trunk.append(output[-1]))
        for mode in args.modes:
            for block, original in zip(reference.trunk.blocks,original_mlps):
                block.mlp.forward = original if mode == "bf16_reference" else lambda x, m=block.mlp: plain_mlp(m,x)
            for batch in ([1,2] if mode == "fp16" else [1]):
                inputs = image if batch == 1 else torch.cat([image,image.flip(-1)],0)
                trunk.clear()
                with torch.autocast(args.device, enabled=mode != "fp32", dtype=torch.float16 if mode == "fp16" else torch.bfloat16):
                    expected = flat_output(reference(inputs),tri)
                expected["trunk"] = trunk.pop()
                # The native guard must restore this outer state even when selecting a different mode.
                outer_dtype = torch.get_autocast_dtype(args.device)
                actual = torch.ops.sam3_native.vision_encode(str(args.store),label,inputs,mode,[])
                assert not torch.is_autocast_enabled(args.device) and torch.get_autocast_dtype(args.device) == outer_dtype
                assert set(actual) == set(expected)
                errors = {}
                for key in expected:
                    assert actual[key].dtype == expected[key].dtype
                    diff = (actual[key].float() - expected[key].float()).abs()
                    errors[key] = {"max_abs":diff.max().item(),"mean_abs":diff.mean().item(),"dtype":str(actual[key].dtype),"shape":list(actual[key].shape)}
                    torch.testing.assert_close(actual[key],expected[key],rtol=2e-5,atol=2e-5)
                result = {"model":label,"mode":mode,"batch":batch,"max_abs":max(x["max_abs"] for x in errors.values()),"outputs":errors}
                results.append(result)
                print(json.dumps({key:value for key,value in result.items() if key != "outputs"}),flush=True)
                del actual,expected
        handle.remove()
        del reference,state,original_mlps
    report = {"device":args.device,"gpu":torch.cuda.get_device_name() if args.device == "cuda" else None,"torch":torch.__version__,"tf32":False,"resolution":1008,"depth":32,
              "rtol":2e-5,"atol":2e-5,"reference_policy":"bf16_reference is unchanged upstream fused MLP; fp32/fp16 replace only the hard-coded BF16 MLP with ordinary Linear/GELU/Linear as in the existing Turing patch.","cases":results}
    args.report.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()

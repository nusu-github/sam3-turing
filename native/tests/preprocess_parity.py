"""Compare native image preprocessing against installed torchvision v2."""
import argparse
import json
from pathlib import Path
import torch
from torchvision.transforms import v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library",type=Path)
    parser.add_argument("--cuda",action="store_true")
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.manual_seed(705)
    torch.set_num_threads(4)
    transform = v2.Compose([v2.ToDtype(torch.uint8,scale=True),v2.Resize((1008,1008)),v2.ToDtype(torch.float32,scale=True),v2.Normalize([0.5]*3,[0.5]*3)])
    results = []
    for device in (["cpu","cuda"] if args.cuda else ["cpu"]):
        for shape in [(3,1,1),(3,19,37),(3,1601,997),(3,1008,1008),(2,3,37,19),(0,3,7,11)]:
            base = torch.randint(0,256,shape,device=device,dtype=torch.uint8)
            for dtype in [torch.uint8,torch.float16,torch.float32,torch.float64,torch.int8,torch.int16,torch.int32,torch.int64,torch.uint16]:
                if dtype.is_floating_point:
                    image = base.to(dtype) / 255
                elif dtype == torch.uint8:
                    image = base
                elif dtype == torch.uint16:
                    image = (base.to(torch.int32)*256).to(dtype)
                else:
                    # Include signed-range and bit-scaling behavior without assuming
                    # that all source integer tensors represent ordinary byte RGB.
                    image = base.to(dtype)
                image = image.transpose(-1,-2)
                expected = transform(image)
                if expected.ndim == 3:
                    expected = expected[None]
                actual = torch.ops.sam3_native.preprocess_rgb(image)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                results.append({"device":device,"shape":list(image.shape),"dtype":str(dtype)})
        # Typical HWC-to-CHW input has channels-last strides after batch insertion.
        image = torch.randint(0,256,(41,53,3),device=device,dtype=torch.uint8).permute(2,0,1)
        actual = torch.ops.sam3_native.preprocess_rgb(image)
        expected = transform(image)[None]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert actual.stride() == expected.stride(), (actual.stride(),expected.stride())
        results.append({"device":device,"case":"channels-last-strides"})
    report = {"exact_cases":len(results),"torch":torch.__version__,"cases":results}
    print(json.dumps({"exact_cases":len(results)}))
    args.report.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()

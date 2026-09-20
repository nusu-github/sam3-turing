"""GPU adaptations of the previously validated CPU research variants.

Only the round-2 output allocation is changed from CPU to the input device.
The group variant already uses device-aware allocations. No source edit is
made to the official SAM3 repository.
"""
import ast
import inspect
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent / 'sam3_cpu_lab'
sys.path.insert(0, str(LAB / 'round2'))
sys.path.insert(0, str(LAB / 'round3'))
from variants import build_head, make_inputs, choose_variant, stream_head
from group_stream import group_stream


def _gpu_stream():
    tree = ast.parse(textwrap.dedent(inspect.getsource(stream_head)))
    class DeviceAllocation(ast.NodeTransformer):
        changed = 0
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'empty':
                node.keywords.append(ast.keyword(arg='device', value=ast.Attribute(value=ast.Name(id='current', ctx=ast.Load()), attr='device', ctx=ast.Load())))
                self.changed += 1
            return node
    transform = DeviceAllocation()
    tree = ast.fix_missing_locations(transform.visit(tree))
    assert transform.changed == 2
    scope = dict(stream_head.__globals__)
    exec(compile(tree, '<device-aware-stream-head>', 'exec'), scope)
    return scope['stream_head']


gpu_stream_head = _gpu_stream()


def inputs_to_device(inputs, device):
    import torch
    result = {}
    for key, value in inputs.items():
        if isinstance(value, list):
            result[key] = [x.to(device) for x in value]
        elif isinstance(value, torch.Tensor):
            # Metadata is read synchronously by prototype guards; keep it on CPU.
            result[key] = value if key == 'image_ids' else value.to(device)
        else:
            result[key] = value
    return result

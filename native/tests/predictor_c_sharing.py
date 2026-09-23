"""Development-only allocator audit of immutable core sharing through the C ABI."""
import argparse
import json
from pathlib import Path

import cffi
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('library', type=Path)
    p.add_argument('store', type=Path)
    p.add_argument('vocabulary', type=Path)
    p.add_argument('--model', type=int, choices=[3, 31], required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    # Parse the public, C-only portion; no private C++ types or native test hooks.
    header = (Path(__file__).parents[1] / 'include/sam3/c_api.h').read_text()
    header = header[header.index('typedef int32_t sam3_status'):header.rindex('#ifdef __cplusplus')]
    ffi = cffi.FFI()
    ffi.cdef('typedef int int32_t; typedef unsigned int uint32_t; typedef long long int64_t; '
             'typedef unsigned long long uint64_t; typedef unsigned char uint8_t;\n' +
             header.replace('SAM3_NATIVE_EXPORT', '').replace('SAM3_NOEXCEPT', ''))
    lib = ffi.dlopen(str(a.library.resolve()))

    def ok(status):
        assert status == 0, ffi.string(lib.sam3_last_error()).decode()

    torch.cuda.init()
    baseline = torch.cuda.memory_allocated()
    config = ffi.new('sam3_context_options*')
    ok(lib.sam3_context_options_init(config))
    strings = [ffi.new('char[]', str(x).encode()) for x in (a.store, a.vocabulary, 'cuda')]
    config.weight_directory, config.vocabulary_path, config.device = strings
    config.model, config.precision = a.model, 2
    context = ffi.new('sam3_context**')
    ok(lib.sam3_context_create(config, context))
    options = ffi.new('sam3_predictor_options*')
    ok(lib.sam3_predictor_options_init(options, a.model))
    options.tracking.frames, options.tracking.height, options.tracking.width = 1, 720, 1280
    provider = ffi.callback('int32_t(void*,int64_t,sam3_rgb_view*)', lambda *_: -1)
    options.tracking.provider = provider
    first, second = ffi.new('sam3_predictor**'), ffi.new('sam3_predictor**')
    ok(lib.sam3_predictor_create(context[0], options, first))
    one = torch.cuda.memory_allocated()
    ok(lib.sam3_predictor_create(context[0], options, second))
    two = torch.cuda.memory_allocated()
    ok(lib.sam3_context_trim(context[0]))
    lib.sam3_context_release(context[0])
    lib.sam3_predictor_release(first[0])
    remaining = torch.cuda.memory_allocated()
    lib.sam3_predictor_release(second[0])
    final = torch.cuda.memory_allocated()
    report = dict(model=a.model, baseline_bytes=baseline, one_owner_bytes=one,
                  two_owner_bytes=two, second_owner_delta_bytes=two-one,
                  surviving_owner_bytes=remaining, after_release_bytes=final,
                  scope='CUDA allocator active bytes during C API creation/destruction, before frame inference. Shared vision/detector/tracker cores only; per-owner features/history and temporary text loads are excluded.')
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert one - baseline > 100_000_000, 'audit did not load real model cores'
    assert two - one < 1_000_000, 'second owner duplicated heavy modules'
    assert remaining >= one, 'destroying one owner released another owner\'s modules'
    assert final == baseline, 'released owners/context retained CUDA allocations'


if __name__ == '__main__':
    main()

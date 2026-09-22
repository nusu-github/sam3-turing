#pragma once
#include "sam3/multiplex_frame.h"
namespace sam3 {
// Only the requested fields are read; missing in-memory tensors are resolved
// through the immutable archive. Returned tensors keep original layouts/devices.
// Confidence is small selection metadata and always stays resident.
constexpr uint32_t history_masks=1,history_spatial=2,history_pointers=4,history_all=7,history_output=8;
SAM3_NATIVE_EXPORT MultiplexFrame load_multiplex_frame(const MultiplexFrame&,uint32_t fields=history_all);
// Preserve all planner metadata; read only selected spatial/pointer streams.
SAM3_NATIVE_EXPORT MultiplexFrameHistory load_selected_multiplex_history(const MultiplexFrameHistory&,
    int64_t frame,int64_t frame_count,bool reverse,const MultiplexTemporalOptions&);
SAM3_NATIVE_EXPORT void archive_multiplex_frame(MultiplexFrame&,const std::filesystem::path&);
}

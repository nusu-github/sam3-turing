#pragma once
#include "sam3/multiplex_frame.h"
#include <functional>
namespace sam3 {
// Dense spatial memory is jointly encoded per bucket, not an object axis.
// Rebuild only when bucket membership/slots change; pointers are remappable.
using MultiplexHistoryRebuilder=std::function<std::pair<at::Tensor,at::Tensor>(const MultiplexFrame&,const MultiplexState&)>;
// States must carry unique global object IDs. New objects have absent masks,
// zero pointers/IoUs, and no conditioning flag in older frames. All changes are
// committed together; an unavailable/failed required rebuild leaves history intact.
// Optional retain() runs per rebuilt frame before commit (e.g. disk paging),
// so remapping need not materialize every frame at once. It must not mutate
// inputs or retain all payloads if bounded memory is required.
SAM3_NATIVE_EXPORT void remap_multiplex_history(MultiplexFrameHistory&,
    const MultiplexState& source,const MultiplexState& destination,
    const MultiplexHistoryRebuilder& rebuild={},const std::string& mode="fp32",
    const std::function<void(MultiplexFrame&)>& retain={});
}

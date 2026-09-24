#pragma once
#include "sam3/occlusion.h"
#include "sam3/multiplex_session.h"
namespace sam3 {
struct ReconditionExecution {
  std::set<int64_t> affected_ids;
  std::vector<int64_t> edited_states,preflight_states;
};
// Execute prepared edits in source order. Sessions are borrowed and must be
// distinct; they share their model cores/feature providers with the video host.
// SAM3 preflights affected states after each candidate. SAM3.1 batches each
// target into its first containing state, then preflights every state sharing
// any affected ID. Complete states, not only edited targets, are affected.
// This mutates sessions. Each SAM3.1 edit is transactional, but an exception
// after an earlier state commits is not an all-session rollback.
SAM3_NATIVE_EXPORT ReconditionExecution execute_reconditioning(int64_t frame,
    const ReconditionMasks&,const std::vector<Sam3TrackingSession*>&);
SAM3_NATIVE_EXPORT ReconditionExecution execute_reconditioning(int64_t frame,
    const ReconditionMasks&,const std::vector<Sam31TrackingSession*>&);
}

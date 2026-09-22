#pragma once
#include "sam3/association.h"
#include <set>
namespace sam3 {
struct HotstartOptions {
  int64_t delay=0,unmatched_threshold=3,duplicate_threshold=3;
  int64_t initial_keep_alive=0,min_keep_alive=-4,max_keep_alive=8;
  bool suppress_only_within_hotstart=true,decrease_for_empty=false;
};
struct HostHotstartState {
  std::map<int64_t,int64_t> first_frame,keep_alive;
  std::map<int64_t,std::vector<int64_t>> unmatched_frames;
  std::map<std::pair<int64_t,int64_t>,std::vector<int64_t>> overlap_frames;
  std::set<int64_t> removed;
  std::map<int64_t,std::set<int64_t>> suppressed;
};
struct HostHotstartResult {HostHotstartState state;std::set<int64_t> newly_removed;};
// Source ID-indexed CPU policy (same state logic in both source classes).
// Frame histories accumulate; they are not consecutive streaks. Input is unchanged.
SAM3_NATIVE_EXPORT HostHotstartResult update_host_hotstart(const HostHotstartState&,
    const AssociationMetadata&,const std::vector<int64_t>& new_ids,int64_t frame,
    bool reverse=false,const HotstartOptions& options={});

struct DeviceHotstartState {
  at::Tensor first_frame,unmatched_count,keep_alive,removed,overlap_count,last_occluded;
  int64_t size() const {return first_frame.size(0);}
};
struct DeviceHotstartResult {DeviceHotstartState state;at::Tensor remove,suppress;};
// SAM3.1 position-indexed policy, distinct from the host policy. Tensor inputs
// are never modified. Unchanged output fields may share storage with inputs.
SAM3_NATIVE_EXPORT DeviceHotstartState empty_device_hotstart(at::Device device=at::kCPU);
SAM3_NATIVE_EXPORT DeviceHotstartResult update_device_hotstart(const DeviceHotstartState&,
    const AssociationTensors&,int64_t frame,bool reverse=false,const HotstartOptions& options={});
SAM3_NATIVE_EXPORT DeviceHotstartState select_device_hotstart(const DeviceHotstartState&,const at::Tensor& indices);
SAM3_NATIVE_EXPORT std::pair<DeviceHotstartState,at::Tensor> compact_device_hotstart(const DeviceHotstartState&);
SAM3_NATIVE_EXPORT DeviceHotstartState extend_device_hotstart(const DeviceHotstartState&,int64_t count,
    int64_t frame,int64_t initial_keep_alive=0);

struct ConfirmationState {std::vector<int64_t> status,consecutive_detections;};
// Status 1=unconfirmed, 2=confirmed. Once confirmed, a missed frame does not undo
// confirmation; it resets only the consecutive detection count. IDs may reorder.
SAM3_NATIVE_EXPORT ConfirmationState update_confirmation(const ConfirmationState&,
    const std::vector<int64_t>& previous_ids,const std::vector<int64_t>& updated_ids,
    const std::map<int64_t,std::vector<int64_t>>& detection_to_tracks,
    const std::vector<int64_t>& new_ids,int64_t threshold=3);
}

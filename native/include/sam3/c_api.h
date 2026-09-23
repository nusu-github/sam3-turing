#ifndef SAM3_C_API_H
#define SAM3_C_API_H
#include <stddef.h>
#include <stdint.h>
#include "sam3_native_export.h"
#ifdef __cplusplus
extern "C" {
#define SAM3_NOEXCEPT noexcept
#else
#define SAM3_NOEXCEPT
#endif

#define SAM3_ABI_VERSION 1u
/* All paths/text are UTF-8; buffers are host memory. No Torch/C++ types occur
 * in this header. Initialise option structs with the matching init function.
 * Sessions are serial; concurrent/reentrant calls return BUSY. Only cancel is
 * allowed while a video call runs. Do not destroy a handle during a call.
 * A context may be released before its children; they retain its modules. */
typedef int32_t sam3_status;
enum {SAM3_OK=0,SAM3_INVALID_ARGUMENT=1,SAM3_RUNTIME_ERROR=2,SAM3_OUT_OF_MEMORY=3,SAM3_BUSY=4,SAM3_CALLBACK_ERROR=5,SAM3_UNSUPPORTED=6};
enum {SAM3_MODEL_3=3,SAM3_MODEL_31=31};
enum {SAM3_FP32=0,SAM3_FP16=1,SAM3_BF16_REFERENCE=2};
enum {SAM3_U8=1,SAM3_BOOL=2,SAM3_I32=3,SAM3_I64=4,SAM3_F16=5,SAM3_BF16=6,SAM3_F32=7,SAM3_F64=8};
enum {SAM3_IMAGE_GROUNDING=1,SAM3_IMAGE_INTERACTIVE=2,SAM3_IMAGE_ALL=3};
typedef struct sam3_context sam3_context;
typedef struct sam3_image sam3_image;
typedef struct sam3_video sam3_video;
typedef struct sam3_predictor sam3_predictor;
typedef struct sam3_media sam3_media;
typedef struct sam3_result sam3_result;
/* Contiguous row-major tensor. Dimensions may be zero. NULL optional view
 * pointers mean absent, not an empty tensor. A nonempty tensor needs data.
 * Input buffers are copied synchronously; output views borrow result storage.
 * F16/BF16 are raw 16-bit encodings, BOOL is one byte per value. */
typedef struct sam3_tensor_view {const void* data;uint64_t bytes;int32_t dtype;int64_t rank;const int64_t* shape;} sam3_tensor_view;
/* Interleaved RGB8, with optional row padding. row_bytes=0 means width*3. */
typedef struct sam3_utf8_view {const char* data;uint64_t bytes;} sam3_utf8_view;
typedef struct sam3_rgb_view {const uint8_t* data;uint64_t bytes;int64_t height,width,row_bytes;} sam3_rgb_view;
typedef struct sam3_context_options {
  uint32_t struct_size,abi_version;
  const char* weight_directory;
  const char* vocabulary_path; /* optional until tokenizing UTF-8 text */
  const char* device; /* "cpu", "cuda", "cuda:N" */
  int32_t model,precision;
} sam3_context_options;
SAM3_NATIVE_EXPORT uint32_t sam3_abi_version(void) SAM3_NOEXCEPT;
/* Thread-local; valid until the next API call on this thread. */
SAM3_NATIVE_EXPORT const char* sam3_last_error(void) SAM3_NOEXCEPT;
/* Process-wide Torch settings: call before concurrent inference/other Torch use. */
SAM3_NATIVE_EXPORT sam3_status sam3_runtime_configure(int32_t cpu_threads,int32_t allow_tf32) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_context_options_init(sam3_context_options*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_context_create(const sam3_context_options*,sam3_context**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT void sam3_context_release(sam3_context*) SAM3_NOEXCEPT;
/* Releases cached modules with no active users; live sessions/results survive. */
SAM3_NATIVE_EXPORT sam3_status sam3_context_trim(sam3_context*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_result_retain(const sam3_result*,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT void sam3_result_release(sam3_result*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_result_field_count(const sam3_result*,int64_t*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_result_field_name(const sam3_result*,int64_t,const char**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_result_get(const sam3_result*,const char*,sam3_tensor_view*) SAM3_NOEXCEPT;
/* tokenize -> tokens[B,32]; encode_text/encode_tokens also return padding[B,L],
 * features[L,B,256], embeddings[L,B,1024]. Original context length is preserved. */
SAM3_NATIVE_EXPORT sam3_status sam3_tokenize(sam3_context*,const char* const* texts,int64_t count,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_encode_text(sam3_context*,const char* const* texts,int64_t count,sam3_result**) SAM3_NOEXCEPT;
/* Counted UTF-8 variants preserve embedded NULs for source text cleaning. */
SAM3_NATIVE_EXPORT sam3_status sam3_tokenize_utf8(sam3_context*,const sam3_utf8_view*,int64_t count,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_encode_text_utf8(sam3_context*,const sam3_utf8_view*,int64_t count,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_encode_tokens(sam3_context*,const sam3_tensor_view*,sam3_result**) SAM3_NOEXCEPT;

SAM3_NATIVE_EXPORT sam3_status sam3_image_create(sam3_context*,uint32_t features,sam3_image**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT void sam3_image_release(sam3_image*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_image_set_rgb(sam3_image*,const sam3_rgb_view*) SAM3_NOEXCEPT;
/* Separate entry preserves the source's single-image vs batch stacking layout. */
SAM3_NATIVE_EXPORT sam3_status sam3_image_set_rgb_batch(sam3_image*,const sam3_rgb_view*,int64_t count) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_image_reset(sam3_image*) SAM3_NOEXCEPT;
typedef struct sam3_interactive_request {
  uint32_t struct_size;int64_t image_index;
  const sam3_tensor_view *points,*labels,*boxes,*masks;
  int32_t pixel_coordinates,multimask,return_logits;
  double mask_threshold,max_hole_area,max_sprinkle_area;
} sam3_interactive_request;
/* Points[N,2] or [B,N,2], labels[N] or [B,N], boxes[4] or [B,4], previous
 * logits[1,288,288] or [B,1,288,288]. Returns masks, iou, low_res_logits. */
SAM3_NATIVE_EXPORT sam3_status sam3_interactive_request_init(sam3_interactive_request*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_image_predict(sam3_image*,const sam3_interactive_request*,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_image_embedding(sam3_image*,sam3_result**) SAM3_NOEXCEPT;
typedef struct sam3_grounding_request {
  uint32_t struct_size;
  const sam3_tensor_view *image_ids,*text_ids,*text_features,*text_padding;
  const sam3_tensor_view *points,*point_labels,*point_padding,*boxes,*box_labels,*box_padding;
  const sam3_tensor_view *visual_features,*visual_padding,*previous_mask;
  int32_t use_text;double confidence_threshold;int64_t resize_chunk;
} sam3_grounding_request;
/* image_ids/text_ids int64[B]. Geometry: normalized xy[N,B,2] / cxcywh[N,B,4],
 * labels int64[N,B] (0/1), right-padding bool[B,N]. Absent geometry is empty.
 * Text/visual/previous-mask layouts match grounding.h. No detection count cap.
 * Returns batch_count and "B/boxes", "B/scores", "B/mask_probabilities",
 * "B/masks", "B/query_indices" (B is a zero-based decimal batch index), plus
 * raw/logits, raw/boxes, raw/masks, raw/semantic and raw/presence_logits. */
SAM3_NATIVE_EXPORT sam3_status sam3_grounding_request_init(sam3_grounding_request*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_image_ground(sam3_image*,const sam3_grounding_request*,sam3_result**) SAM3_NOEXCEPT;

/* Provider returns 0 on success. Its pixel buffer must remain valid until the
 * next provider call or completion of the enclosing API call. It may reuse one
 * buffer. Any nonzero return aborts with CALLBACK_ERROR; no Python is involved. */
typedef int32_t (*sam3_frame_provider)(void*,int64_t,sam3_rgb_view*);
/* Output callback borrows a result; retain it for use afterward. Return 0 to
 * continue, 1 to stop consistently after this frame, -1 to report an error.
 * Video outputs contain frame, ids, masks, low_masks (when available), logits. */
typedef int32_t (*sam3_output_callback)(void*,const sam3_result*);
typedef struct sam3_video_options {
  uint32_t struct_size;int64_t frames,height,width;
  sam3_frame_provider provider;void* provider_user;
  int32_t offload_state,non_overlap_output,non_overlap_memory;
  int32_t all_edits_conditioning,always_start_at_first_annotation;
  int32_t clear_near_input,clear_near_multi_object;
  int64_t fill_hole_area,memory_slots,max_conditioning_frames,max_pointer_frames,stride;
  int32_t keep_first,select_by_score;double score_threshold;
  int32_t multimask,multimask_tracking;int64_t multimask_min_points,multimask_max_points;
  /* SAM3.1-specific fields; history_directory pages retained frame payloads. */
  const char* history_directory;
  int32_t attenuate_iou_by_stability,only_past_pointers,signed_pointer_time,temporal_v2,encode_pointer_time,use_pointers;
  double object_threshold;
} sam3_video_options;
SAM3_NATIVE_EXPORT sam3_status sam3_video_options_init(sam3_video_options*,int32_t model) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_video_create(sam3_context*,const sam3_video_options*,sam3_video**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT void sam3_video_release(sam3_video*) SAM3_NOEXCEPT;
/* Coordinates normalized [0,1] when normalized=1, model-space otherwise.
 * points/labels describe one object; box optional [4]. No point count cap. */
SAM3_NATIVE_EXPORT sam3_status sam3_video_add_points(sam3_video*,int64_t frame,int64_t object,
    const sam3_tensor_view* points,const sam3_tensor_view* labels,const sam3_tensor_view* box,
    int32_t normalized,int32_t clear_old,int32_t use_previous,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_video_add_mask(sam3_video*,int64_t frame,int64_t object,const sam3_tensor_view*,sam3_result**) SAM3_NOEXCEPT;
/* Simultaneous brushes, SAM3.1 only; ids[count], masks[count,H,W]. */
SAM3_NATIVE_EXPORT sam3_status sam3_video_add_masks(sam3_video*,int64_t frame,const int64_t* ids,int64_t count,const sam3_tensor_view*,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_video_preflight(sam3_video*,int32_t encode_memory) SAM3_NOEXCEPT;
/* start/max_steps=-1 use source defaults; max_steps=0 emits the start frame. */
SAM3_NATIVE_EXPORT sam3_status sam3_video_propagate(sam3_video*,int64_t start,int64_t max_steps,int32_t reverse,int32_t encode_memory,int32_t preflight,sam3_output_callback,void*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_video_cancel(sam3_video*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_video_clear_input(sam3_video*,int64_t frame,int64_t object,sam3_result**) SAM3_NOEXCEPT;
/* SAM3 removal previews use callback if supplied; SAM3.1 emits none. */
SAM3_NATIVE_EXPORT sam3_status sam3_video_remove_object(sam3_video*,int64_t object,int32_t strict,sam3_output_callback,void*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_video_object_ids(sam3_video*,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_video_reset(sam3_video*) SAM3_NOEXCEPT;

/* Owning semantic image/video predictor. Distinct from sam3_video's low-level
 * tracker: owns detection, IDs, edit routing, output buffering and image policy.
 * tracking supplies source dimensions/provider and temporal tracker options.
 * Initialize with the context's model; create rejects a model mismatch. */
typedef struct sam3_predictor_options {
  uint32_t struct_size;int32_t model;
  sam3_video_options tracking;
  int32_t centers,image_only;int64_t output_batch_size;
  double image_detection_threshold;
  int32_t nms; /* 0 greedy, 1 SAM3.1 perflib, 2 SAM3.1 batched */
  double detection_score_threshold,nms_threshold,detection_boundary_margin;
  int32_t detection_use_iom,detection_boundary_filter;
  /* Counts use FP32 by default; FP16 requests source overflow compatibility. */
  int32_t policy_precision;
  double new_detection_threshold,track_match_threshold,detection_match_threshold;
  double high_confidence_threshold,iom_recondition_threshold,iou_recondition_threshold;
  int32_t association_use_iom;int64_t pad_tracks_to;
  int64_t hotstart_delay,unmatched_threshold,duplicate_threshold;
  int64_t initial_keep_alive,min_keep_alive,max_keep_alive;
  int32_t suppress_only_within_hotstart,decrease_for_empty;
  int64_t recondition_period;
  double recondition_box_iou_threshold,recondition_detection_score_threshold;
  int32_t boundary_filter,confirmation_enabled,warmup_complete;
  int32_t allow_unoccluded_suppression,reapply_no_object_pointer;
  double boundary_margin,occlusion_threshold;
  int64_t confirmation_threshold,cleanup_area,bucket_capacity;
} sam3_predictor_options;
typedef struct sam3_semantic_prompt {
  uint32_t struct_size;
  const sam3_utf8_view* text; /* NULL=absent, non-NULL empty text is a prompt */
  const sam3_tensor_view *boxes_xywh,*box_labels; /* normalized [N,4], int64[N] */
  const sam3_tensor_view *visual_features,*visual_padding; /* [N,1,256], bool[1,N] */
} sam3_semantic_prompt;
typedef enum sam3_video_preprocess {
  SAM3_PREPROCESS_IMAGE_FOLDER=0, SAM3_PREPROCESS_PIL_LIST=1,
  SAM3_PREPROCESS_TORCHCODEC_CPU=2, SAM3_PREPROCESS_TORCHCODEC_CUDA=3,
  SAM3_PREPROCESS_CV2_SOURCE=4
} sam3_video_preprocess;
/* These select preprocessing only; decoding and image/video semantics stay
 * explicit. Cv2Source preserves upstream's no-/255 behavior. */
SAM3_NATIVE_EXPORT int32_t sam3_video_preprocess_available(int32_t policy) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_preprocess_video_rgb(const sam3_rgb_view*,int32_t policy,const char* device,sam3_result**) SAM3_NOEXCEPT;
/* Before first encoding or after reset. Default remains IMAGE_FOLDER. */
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_set_preprocess(sam3_predictor*,int32_t policy) SAM3_NOEXCEPT;
/* Before use or after reset. Nonempty CPU/CUDA device list, copied on return.
 * The context device remains the vision/detection/output coordinator. Duplicate
 * devices represent logical ranks and share cores. Existing ABI structs do not
 * change. Execution is synchronous; cancel remains callable from another thread. */
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_set_tracking_devices(sam3_predictor*,const char* const* devices,int64_t count) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_options_init(sam3_predictor_options*,int32_t model) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_semantic_prompt_init(sam3_semantic_prompt*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_create(sam3_context*,const sam3_predictor_options*,sam3_predictor**) SAM3_NOEXCEPT;
/* Optional native media support; existing ABI 1 structures remain unchanged.
 * Paths are UTF-8 local files/folders. read_frame returns owning rgb [3,H,W]
 * plus seconds/duration scalar fields. Results outlive the media handle.
 * Opening video scans decoded timestamps to count frames exactly, including
 * delayed frames; random/reverse reads currently replay from the beginning.
 * create_from_media ignores tracking provider/dimensions and retains the source.
 * image_only remains an explicit predictor option (one-frame video is distinct).
 */
typedef struct sam3_media_options {uint32_t struct_size;int32_t image_only,threads;} sam3_media_options;
SAM3_NATIVE_EXPORT int32_t sam3_media_available(void) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_media_options_init(sam3_media_options*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_media_open(const char* path,const sam3_media_options*,sam3_media**) SAM3_NOEXCEPT;
enum {SAM3_MEDIA_COLOR_STREAM=0,SAM3_MEDIA_COLOR_OPENCV=1};
/* Immutable per-source video color conversion. OpenCV emulates its FFmpeg
 * BGR24/default-colorspace conversion, then returns RGB. Images stay Pillow-like. */
SAM3_NATIVE_EXPORT sam3_status sam3_media_open_with_color(const char* path,const sam3_media_options*,int32_t color_policy,sam3_media**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT void sam3_media_release(sam3_media*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_media_info(sam3_media*,sam3_result**) SAM3_NOEXCEPT;
/* Statistics are owning scalar result fields. Budget excludes decoder memory and
 * the last-frame cache; zero disables the additional RGB window. ABI1 additive. */
SAM3_NATIVE_EXPORT sam3_status sam3_media_stats(sam3_media*,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_media_set_cache_bytes(sam3_media*,int64_t bytes) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_media_read_frame(sam3_media*,int64_t frame,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_media_write_png(const char* path,const sam3_rgb_view*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_create_from_media(sam3_context*,sam3_media*,const sam3_predictor_options*,sam3_predictor**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT void sam3_predictor_release(sam3_predictor*) SAM3_NOEXCEPT;
/* Results: frame, ids, probabilities, boxes_xywh, bool masks[N,H,W], optional
 * normalized centers[N,2]; cached_ids and cached_masks/ID retain pre-overlap
 * bool[1,H,W] inputs including empty masks. Optional stats/NAME are int64.
 * IDs/probabilities/boxes/masks follow the same order. Views borrow the result.
 * Semantic replacement resets observations; execution errors are not a rollback. */
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_add_prompt(sam3_predictor*,int64_t frame,const sam3_semantic_prompt*,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_add_points(sam3_predictor*,int64_t frame,int64_t object,
    const sam3_tensor_view* points,const sam3_tensor_view* labels,const sam3_tensor_view* box,
    int32_t normalized,int32_t clear_old,int32_t use_previous,int32_t stateless,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_add_mask(sam3_predictor*,int64_t frame,int64_t object,const sam3_tensor_view*,sam3_result**) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_fetch(sam3_predictor*,int64_t frame,sam3_result**) SAM3_NOEXCEPT;
/* start/steps=-1 select source defaults. Forward includes start, reverse starts
 * at start-1 (unlike low-level tracker propagation). force_tracker is SAM3-only.
 * Callbacks may retain results; return 0=continue, 1=cancel, other=error.
 * Cancellation stops buffered emissions too, without interrupting a CUDA kernel. */
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_propagate(sam3_predictor*,int64_t start,int64_t steps,int32_t reverse,int32_t force_tracker,sam3_output_callback,void*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_cancel(sam3_predictor*) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_remove_object(sam3_predictor*,int64_t object) SAM3_NOEXCEPT;
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_reset(sam3_predictor*) SAM3_NOEXCEPT;
/* Diagnostic fields: ids, visual_encodes, cached_frames, action_count. */
SAM3_NATIVE_EXPORT sam3_status sam3_predictor_info(sam3_predictor*,sam3_result**) SAM3_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef SAM3_NOEXCEPT
#endif

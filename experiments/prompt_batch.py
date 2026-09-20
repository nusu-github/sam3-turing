"""Experimental text-prompt batching over a single cached image."""

from dataclasses import replace

import torch

from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from sam3.turing import _compile_text_grounding


class PromptBatch:
    def __init__(self, processor):
        self.processor = processor
        self.grounding = _compile_text_grounding(processor.model, "reduce-overhead")
        self.stages = {}

    @torch.inference_mode()
    def __call__(self, captions, state):
        processor = self.processor
        model = processor.model
        count = len(captions)
        if count not in self.stages:
            find = replace(
                processor.find_stage,
                img_ids=torch.zeros(count, device=processor.device, dtype=torch.long),
                text_ids=torch.arange(count, device=processor.device),
            )
            self.stages[count] = (find, model._get_dummy_prompt(count))
        find, geometry = self.stages[count]
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            backbone = dict(state["backbone_out"])
            backbone.update(
                model.backbone.forward_text(captions, device=processor.device)
            )
            output = self.grounding(
                backbone_out=backbone,
                find_input=find,
                geometric_prompt=geometry,
                find_target=None,
            )
            probabilities = (
                output["pred_logits"].sigmoid()
                * output["presence_logit_dec"].sigmoid().unsqueeze(1)
            ).squeeze(-1)
            h, w = state["original_height"], state["original_width"]
            scale = torch.tensor([w, h, w, h], device=processor.device)[None]
            results = []
            for index in range(count):
                keep = probabilities[index] > processor.confidence_threshold
                masks = interpolate(
                    output["pred_masks"][index, keep].unsqueeze(1),
                    (h, w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid_()
                results.append(
                    {
                        "scores": probabilities[index, keep],
                        "boxes": box_ops.box_cxcywh_to_xyxy(
                            output["pred_boxes"][index, keep]
                        )
                        * scale,
                        "masks_logits": masks,
                        "masks": masks > 0.5,
                    }
                )
        return results

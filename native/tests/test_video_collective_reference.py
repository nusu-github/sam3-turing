"""Integrity tests for the test-only original-source collection replay."""
import types
import unittest

import numpy as np
import torch

from sam3.model.sam3_video_base import Sam3VideoBase
from video_collective_reference import install_empty_peer_replay


class CollectionReplayTest(unittest.TestCase):
    def model(self, ids, masks, scores):
        host = types.SimpleNamespace(rank=0, world_size=1)
        host._propogate_tracker_one_frame_local_gpu = lambda *a, **k: (ids, masks, scores)
        host.run_tracker_propagation = types.MethodType(Sam3VideoBase.run_tracker_propagation, host)
        return host

    def request(self, host, ids):
        return host.run_tracker_propagation(
            frame_idx=1, num_frames=4, reverse=False, tracker_states_local=[],
            tracker_metadata_prev=dict(obj_ids_per_gpu=[ids], num_obj_per_gpu=[len(ids)]),
        )

    def test_original_method_and_distributed_globals_are_unchanged(self):
        method = Sam3VideoBase.run_tracker_propagation
        code, dist, gather = method.__code__, method.__globals__["dist"], torch.distributed.all_gather
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for count in (0, 2):
                    for ranks in (2, 3):
                        with self.subTest(device=device, dtype=dtype, count=count, ranks=ranks):
                            ids = np.array([-9, 2**62], np.int64)[:count]
                            masks = torch.arange(count * 15, device=device, dtype=dtype).reshape(count, 5, 3).transpose(1, 2)
                            scores = torch.arange(count, device=device, dtype=dtype)
                            before = masks.clone()
                            host = self.model(ids, masks, scores)
                            records = install_empty_peer_replay(host, ranks)
                            actual = self.request(host, ids)
                            self.assertTrue(torch.equal(actual[0], masks.float()))
                            self.assertTrue(torch.equal(actual[1], scores.float()))
                            self.assertEqual(actual[0].dtype, torch.float32)
                            self.assertTrue(actual[0].is_contiguous())
                            actual[0].zero_()
                            self.assertTrue(torch.equal(masks, before))
                            self.assertEqual(records[0]["all_gather_calls"], ranks * 2)
        self.assertIs(method.__code__, code)
        self.assertIs(method.__globals__["dist"], dist)
        self.assertIs(torch.distributed.all_gather, gather)

    def test_original_id_alignment_assertion_is_retained(self):
        host = self.model(np.array([7]), torch.ones(1, 3, 5), torch.ones(1))
        install_empty_peer_replay(host)
        with self.assertRaises(AssertionError):
            self.request(host, np.array([8]))

    def test_invalid_rank_count_does_not_replace_method(self):
        host = self.model(np.empty(0, np.int64), torch.empty(0, 3, 5), torch.empty(0))
        original = host.run_tracker_propagation
        with self.assertRaises(ValueError):
            install_empty_peer_replay(host, 1)
        self.assertIs(host.run_tracker_propagation, original)


if __name__ == "__main__":
    unittest.main()

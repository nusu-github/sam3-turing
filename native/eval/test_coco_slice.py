"""Evaluator integrity: asymmetric masks, corrupt outputs, known perfect AP."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from evaluate_coco_slice import evaluate, read_case


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = self.root / "7-1.json"
        self.masks = np.zeros((200, 3, 5), dtype=np.uint8)
        self.masks[0, :2, 1:4] = 1
        self.data = dict(
            image_id=7,
            category_id=1,
            height=3,
            width=5,
            mask_bytes=2,
            detections=[
                dict(
                    query=i,
                    score=float(i == 0),
                    box=[1, 0, 4, 2] if i == 0 else [0, 0, 0, 0],
                )
                for i in range(200)
            ],
        )
        self.save()
        np.packbits(self.masks.reshape(200, -1), axis=1, bitorder="little").tofile(
            self.case.with_suffix(".masks.bin")
        )

    def save(self):
        self.case.write_text(json.dumps(self.data))

    def test_asymmetric_non_byte_aligned_masks(self):
        _, decoded = read_case(self.case)
        np.testing.assert_array_equal(decoded, self.masks)
        rle = mask_utils.encode(np.asfortranarray(decoded[0]))
        np.testing.assert_array_equal(mask_utils.decode(rle), self.masks[0])
        self.assertEqual(int(mask_utils.area(rle)), 6)

    def test_truncated_payload(self):
        self.case.with_suffix(".masks.bin").write_bytes(b"\0")
        with self.assertRaisesRegex(ValueError, "payload size"):
            read_case(self.case)

    def test_nonfinite_and_duplicate_query(self):
        self.data["detections"][0]["score"] = float("nan")
        self.save()
        with self.assertRaisesRegex(ValueError, "invalid score"):
            read_case(self.case)
        self.data["detections"][0]["score"] = 1
        self.data["detections"][0]["query"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate queries"):
            read_case(self.case)

    def test_known_perfect_coco_result(self):
        rle = mask_utils.encode(np.asfortranarray(self.masks[0]))
        rle["counts"] = rle["counts"].decode("ascii")
        gt = dict(
            info={},
            images=[dict(id=7, height=3, width=5)],
            categories=[dict(id=1, name="object")],
            annotations=[
                dict(
                    id=1,
                    image_id=7,
                    category_id=1,
                    iscrowd=0,
                    area=6,
                    bbox=[1, 0, 3, 2],
                    segmentation=rle,
                )
            ],
        )
        annotation = self.root / "gt.json"
        annotation.write_text(json.dumps(gt))
        manifest = self.root / "manifest.tsv"
        manifest.write_text("7\t1\tunused\tunused\n")
        (self.root / "metrics.jsonl").write_text(
            json.dumps(
                dict(
                    image_id=7,
                    category_id=1,
                    finite=True,
                    queries=200,
                    image_seconds=1,
                    detector_postprocess_seconds=1,
                    peak_allocated_bytes=0,
                    peak_reserved_bytes=0,
                )
            )
        )
        (self.root / "summary.json").write_text(json.dumps(dict(cases=1, threshold=-1)))
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluate(annotation, manifest, self.root)
        self.assertEqual(report["candidates"], 200)
        for kind in ("bbox", "segm"):
            self.assertAlmostEqual(report["quality"][kind]["AP"], 1.0)
            self.assertAlmostEqual(report["quality"][kind]["AP50"], 1.0)
            self.assertIsNone(report["quality"][kind]["AP_large"])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from compare_image_runs import compare


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.left, self.right = root / "left", root / "right"
        self.manifest = root / "manifest.tsv"
        self.manifest.write_text("7\t1\tunused\tunused\n")
        data = dict(
            image_id=7,
            category_id=1,
            height=3,
            width=5,
            mask_bytes=2,
            probability_dtype="Float",
            probability_bytes=12000,
            detections=[dict(query=i, score=0, box=[0, 0, 0, 0]) for i in range(200)],
        )
        for path in (self.left, self.right):
            path.mkdir()
            (path / "7-1.json").write_text(json.dumps(data))
            (path / "7-1.masks.bin").write_bytes(bytes(400))
            (path / "7-1.probabilities.bin").write_bytes(bytes(12000))

    def test_complete_probability_result(self):
        report = compare(self.manifest, self.left, self.right)
        self.assertEqual(report["full_probability_cases"], 1)
        self.assertEqual(report["candidates"], 200)
        self.assertEqual(len(report["files"]), 3)
        self.assertTrue(report["all_bytes_equal"])

    def test_single_changed_bit(self):
        (self.right / "7-1.probabilities.bin").write_bytes(b"\1" + bytes(11999))
        with self.assertRaisesRegex(ValueError, "output differs"):
            compare(self.manifest, self.left, self.right)

    def test_both_sides_truncated(self):
        for path in (self.left, self.right):
            (path / "7-1.masks.bin").write_bytes(bytes(399))
        with self.assertRaisesRegex(ValueError, "incomplete mask tensor"):
            compare(self.manifest, self.left, self.right)

    def test_missing_case(self):
        (self.right / "7-1.json").unlink()
        with self.assertRaises(FileNotFoundError):
            compare(self.manifest, self.left, self.right)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from src.utils.video_io import VideoReaderCache


class VideoIoTest(unittest.TestCase):
    def test_detect_lfs_pointer_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "file.mp4"
            p.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:dummy\n"
                "size 123\n",
                encoding="utf-8",
            )
            cache = VideoReaderCache.__new__(VideoReaderCache)
            self.assertTrue(cache._is_lfs_pointer_file(p))

    def test_non_lfs_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "file.mp4"
            p.write_bytes(b"\x00\x00\x00\x18ftypisom")
            cache = VideoReaderCache.__new__(VideoReaderCache)
            self.assertFalse(cache._is_lfs_pointer_file(p))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Unit and integration tests for the ETH Zurich Plant Stress Phenotyping Dataset Downloader.
"""

import io
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from download_dataset import (
    EXCLUDED_FILES,
    FileMetadata,
    SafeZipExtractor,
    SecurityError,
    compute_file_md5,
    format_bytes,
    is_html_content,
    ManifestManager,
)


class TestDownloaderUtils(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_format_bytes(self):
        self.assertIn("B", format_bytes(500))
        self.assertIn("KB", format_bytes(2048))
        self.assertIn("MB", format_bytes(5 * 1024 * 1024))
        self.assertIn("GB", format_bytes(3 * 1024 * 1024 * 1024))

    def test_compute_file_md5(self):
        test_file = self.test_dir / "sample.txt"
        test_file.write_text("Hello ETH Zurich!", encoding="utf-8")
        # MD5 of "Hello ETH Zurich!" is 878e4d2938a3cae0f21422fa70691ef3
        import hashlib
        expected_md5 = hashlib.md5("Hello ETH Zurich!".encode("utf-8")).hexdigest()
        self.assertEqual(compute_file_md5(test_file), expected_md5)

    def test_is_html_content(self):
        html_sample = b"<!DOCTYPE html><html><head><title>Too Many Requests</title></head></html>"
        json_sample = b'{"status": 200, "message": "OK"}'
        zip_sample = b"PK\x03\x04\x14\x00\x00\x00\x08\x00"

        self.assertTrue(is_html_content(html_sample))
        self.assertFalse(is_html_content(json_sample))
        self.assertFalse(is_html_content(zip_sample))

    def test_zip_slip_protection(self):
        """Verify that SafeZipExtractor rejects archives containing directory traversal attacks."""
        malicious_zip = self.test_dir / "malicious.zip"
        with zipfile.ZipFile(malicious_zip, "w") as zf:
            # Add an entry attempting to escape to parent directory
            zf.writestr("../evil.txt", "MALICIOUS PAYLOAD")

        extractor = SafeZipExtractor(extracted_dir=self.test_dir / "extracted")
        with self.assertRaises(SecurityError):
            extractor.extract_archive(malicious_zip)

    def test_safe_zip_extraction(self):
        """Verify normal zip extraction succeeds and preserves subdirectories."""
        valid_zip = self.test_dir / "calibration.zip"
        with zipfile.ZipFile(valid_zip, "w") as zf:
            zf.writestr("cam1/intrinsics.txt", "fx: 1000\nfy: 1000")
            zf.writestr("cam2/intrinsics.txt", "fx: 1050\nfy: 1050")

        extractor = SafeZipExtractor(extracted_dir=self.test_dir / "extracted")
        extracted = extractor.extract_archive(valid_zip)
        self.assertEqual(len(extracted), 2)
        target_file = self.test_dir / "extracted" / "calibration" / "cam1" / "intrinsics.txt"
        self.assertTrue(target_file.exists())
        self.assertEqual(target_file.read_text(), "fx: 1000\nfy: 1000")

    def test_images_zip_never_extracted(self):
        """Verify that images.zip is strictly refused from extraction."""
        fake_images_zip = self.test_dir / "images.zip"
        with zipfile.ZipFile(fake_images_zip, "w") as zf:
            zf.writestr("img001.png", "fake image data")

        extractor = SafeZipExtractor(extracted_dir=self.test_dir / "extracted")
        result = extractor.extract_archive(fake_images_zip)
        self.assertEqual(result, [])
        extracted_target = self.test_dir / "extracted" / "images"
        self.assertFalse(extracted_target.exists())

    def test_manifest_manager_excludes_images_zip(self):
        """Verify manifest.json marks images.zip as excluded_already_downloaded."""
        manifest_path = self.test_dir / "plant_stress_dataset" / "manifest.json"
        manifest_mgr = ManifestManager(manifest_path)

        files = {
            "images.zip": FileMetadata(
                filename="images.zip",
                url="https://example.com/images.zip",
                expected_size=2500166272,
                expected_checksum="073b0b873a3751f6f6068905c1816eb8",
                status="excluded_already_downloaded",
            ),
            "calibration.zip": FileMetadata(
                filename="calibration.zip",
                url="https://example.com/calibration.zip",
                expected_size=95976039,
                expected_checksum="f69978b9bae3ac97f6e387458cc35772",
                status="verified",
                actual_size=95976039,
                actual_checksum="f69978b9bae3ac97f6e387458cc35772",
            ),
        }

        manifest_mgr.save_manifest(files, {"title": "Test Dataset"}, self.test_dir)
        self.assertTrue(manifest_path.exists())

        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("images.zip", data["files"])
        self.assertEqual(data["files"]["images.zip"]["status"], "excluded_already_downloaded")
        self.assertEqual(data["files"]["calibration.zip"]["status"], "verified")


if __name__ == "__main__":
    unittest.main()

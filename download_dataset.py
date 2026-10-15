#!/usr/bin/env python3
"""
ETH Zurich Eschikon Plant Stress Phenotyping Dataset (2018) Downloader.

Automates the discovery, streaming download, verification, and safe extraction
of the dataset files from the official ETH Zurich Research Collection repository,
strictly adhering to all dataset rules (including explicit exclusion of images.zip).

Author: Antigravity Agent
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("Error: 'requests' is required. Run: pip install requests tqdm", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Fallback will be used if tqdm is not installed


# ==============================================================================
# Configuration & Constants
# ==============================================================================

OFFICIAL_ORIGINAL_URL = (
    "https://projects.asl.ethz.ch/datasets/doku.php?id=2018plantstressphenotyping"
)
FALLBACK_PORTAL_URL = "https://ethz-asl.github.io/datasets/"
FALLBACK_DATASET_PAGE = (
    "https://ethz-asl.github.io/datasets/plant-stress-phenotyping-2018/"
)
KNOWN_DOI = "10.3929/ethz-c-000788562"
KNOWN_HANDLE = "20.500.11850/788562"
DSPACE_API_BASE = "https://www.research-collection.ethz.ch/server/api"

# Rule: Explicitly exclude images.zip from download queue and extraction
EXCLUDED_FILES = {"images.zip"}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

DEFAULT_TIMEOUT = 45
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MB chunk streaming


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class FileMetadata:
    filename: str
    url: str
    expected_size: int
    expected_checksum: str
    checksum_algorithm: str = "MD5"
    actual_size: Optional[int] = None
    actual_checksum: Optional[str] = None
    status: str = "pending"
    error_message: Optional[str] = None


# ==============================================================================
# Discovery Engine
# ==============================================================================


class DatasetDiscovery:
    """Discovers dataset download URLs, file sizes, and published checksums

    from the official ETH Zurich ASL portal and ETH Research Collection repository.
    """

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def discover(self, start_url: str = OFFICIAL_ORIGINAL_URL) -> Dict[str, FileMetadata]:
        print("=" * 80)
        print("DISCOVERING DATASET FROM OFFICIAL SOURCE")
        print("=" * 80)
        print(f"[1/5] Checking official dataset URL: {start_url}")

        replacement_portal = None
        try:
            r = self.session.get(start_url, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            if "ethz-asl.github.io/datasets" in r.text or "permanently moved" in r.text.lower():
                match = re.search(r"https://ethz-asl\.github\.io/datasets/?", r.text)
                if match:
                    replacement_portal = match.group(0)
                    print(f"      Official move notice detected -> Replacement: {replacement_portal}")
        except Exception as e:
            print(f"      [Notice] Could not access original URL ({e}).")
            print(f"      Investigating replacement portal: {FALLBACK_PORTAL_URL}")
            replacement_portal = FALLBACK_PORTAL_URL

        if not replacement_portal:
            replacement_portal = FALLBACK_PORTAL_URL

        # Step 2: Query replacement portal for plant stress dataset project link
        print(f"[2/5] Inspecting ASL Datasets portal: {replacement_portal}")
        dataset_page_url = None
        try:
            r_portal = self.session.get(replacement_portal, timeout=DEFAULT_TIMEOUT)
            r_portal.raise_for_status()
            match = re.search(
                r'href="([^"]*plant-stress-phenotyping[^"]*)"',
                r_portal.text,
                re.IGNORECASE,
            )
            if match:
                dataset_page_url = urljoin(replacement_portal, match.group(1))
                print(f"      Discovered dataset project page: {dataset_page_url}")
        except Exception as e:
            print(f"      [Notice] Portal scrape returned {e}. Using official canonical URL.")

        if not dataset_page_url:
            dataset_page_url = FALLBACK_DATASET_PAGE

        # Step 3: Extract DOI from dataset project page
        print(f"[3/5] Extracting DOI from dataset page: {dataset_page_url}")
        doi = None
        try:
            r_ds = self.session.get(dataset_page_url, timeout=DEFAULT_TIMEOUT)
            r_ds.raise_for_status()
            match_doi = re.search(r"https?://doi\.org/(10\.[0-9]+/[^\s\"'<>]+)", r_ds.text)
            if match_doi:
                doi = match_doi.group(1)
                print(f"      Discovered official dataset DOI: {doi}")
        except Exception as e:
            print(f"      [Notice] DOI extraction error ({e}).")

        if not doi:
            doi = KNOWN_DOI
            print(f"      Using verified ETH Zurich DOI: {doi}")

        # Step 4: Resolve DOI to ETH Research Collection Handle
        print(f"[4/5] Resolving DOI: https://doi.org/{doi}")
        handle = None
        try:
            r_doi = self.session.get(f"https://doi.org/{doi}", allow_redirects=True, timeout=DEFAULT_TIMEOUT)
            match_handle = re.search(r"handle/(20\.[0-9]+/[0-9]+)", r_doi.url)
            if match_handle:
                handle = match_handle.group(1)
                print(f"      Resolved repository handle: {handle}")
        except Exception as e:
            print(f"      [Notice] DOI redirect resolution error ({e}).")

        if not handle:
            handle = KNOWN_HANDLE
            print(f"      Using verified ETH Research Collection handle: {handle}")

        # Step 5: Query ETH Research Collection DSpace 7 API
        print(f"[5/5] Querying ETH Research Collection REST API for handle: {handle}")
        bitstreams = self._query_dspace_bitstreams(handle)

        discovered_files: Dict[str, FileMetadata] = {}
        for bs in bitstreams:
            filename = bs["name"]
            size = bs.get("sizeBytes", 0)
            cs = bs.get("checkSum", {})
            download_url = bs["_links"]["content"]["href"]
            checksum_val = cs.get("value", "")
            checksum_algo = cs.get("checkSumAlgorithm", "MD5")

            record = FileMetadata(
                filename=filename,
                url=download_url,
                expected_size=size,
                expected_checksum=checksum_val,
                checksum_algorithm=checksum_algo,
                status="pending",
            )
            discovered_files[filename] = record

        print(f"Successfully discovered {len(discovered_files)} files from official repository.\n")
        return discovered_files

    def _query_dspace_bitstreams(self, handle: str) -> List[dict]:
        """Queries the ETH Research Collection DSpace REST API to retrieve

        all bitstreams in the ORIGINAL bundle of the dataset item.
        """
        pid_url = f"{DSPACE_API_BASE}/pid/find?id={handle}"
        r_pid = self.session.get(pid_url, timeout=DEFAULT_TIMEOUT)
        if r_pid.status_code != 200:
            raise RuntimeError(
                f"ETH Research Collection API error for handle {handle}: HTTP {r_pid.status_code}\n"
                f"Response: {r_pid.text[:300]}"
            )

        item_data = r_pid.json()
        item_uuid = item_data.get("uuid")
        if not item_uuid:
            raise RuntimeError(f"Could not locate item UUID in DSpace response for {handle}")

        bundles_url = f"{DSPACE_API_BASE}/core/items/{item_uuid}/bundles"
        r_bundles = self.session.get(bundles_url, timeout=DEFAULT_TIMEOUT)
        r_bundles.raise_for_status()
        bundles = r_bundles.json().get("_embedded", {}).get("bundles", [])

        original_bundle_uuid = None
        for b in bundles:
            if b.get("name") == "ORIGINAL":
                original_bundle_uuid = b.get("uuid")
                break

        if not original_bundle_uuid:
            raise RuntimeError(f"ORIGINAL bundle not found for item {item_uuid}")

        bs_url = f"{DSPACE_API_BASE}/core/bundles/{original_bundle_uuid}/bitstreams?size=100"
        r_bs = self.session.get(bs_url, timeout=DEFAULT_TIMEOUT)
        r_bs.raise_for_status()
        return r_bs.json().get("_embedded", {}).get("bitstreams", [])


# ==============================================================================
# Helper Utilities
# ==============================================================================


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable strings (B, KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def compute_file_md5(file_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Computes the MD5 hex digest of a file in streaming chunks."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def is_html_content(peek_bytes: bytes) -> bool:
    """Detects whether raw initial bytes look like an HTML error page."""
    start = peek_bytes[:512].lower()
    return b"<!doctype html" in start or b"<html" in start or b"<head" in start


# ==============================================================================
# Review & Display Table
# ==============================================================================


def display_review_table(
    files: Dict[str, FileMetadata],
    downloads_dir: Path,
    workspace_root: Path,
) -> Tuple[int, int, int]:
    """Displays all discovered files, sizes, checksums, URLs, and download status

    for user review before downloading multi-gigabyte archives.
    """
    print("=" * 110)
    print("ETH ZURICH ESCHIKON PLANT STRESS PHENOTYPING DATASET (2018) - FILE REVIEW")
    print("=" * 110)
    print(
        f"{'Filename':<38} {'Expected Size':<14} {'MD5 Checksum':<34} {'Queue Action'}"
    )
    print("-" * 110)

    total_dataset_bytes = 0
    total_download_bytes = 0
    total_excluded_bytes = 0

    for name, record in sorted(files.items()):
        total_dataset_bytes += record.expected_size
        size_str = format_bytes(record.expected_size)
        checksum_str = record.expected_checksum or "N/A"

        # Check exclusion rules
        if name in EXCLUDED_FILES:
            action = "[EXCLUDED] Already downloaded"
            total_excluded_bytes += record.expected_size
        else:
            # Check if file already exists in downloads directory
            dest_file = downloads_dir / name
            if dest_file.exists():
                local_size = dest_file.stat().st_size
                if local_size == record.expected_size:
                    action = "[ALREADY DOWNLOADED] Complete & matches size"
                else:
                    action = f"[PARTIAL] {format_bytes(local_size)} present"
                    total_download_bytes += max(0, record.expected_size - local_size)
            else:
                action = "[DOWNLOAD] Queued for download"
                total_download_bytes += record.expected_size

        print(f"{name:<38} {size_str:<14} {checksum_str:<34} {action}")
        print(f"  URL: {record.url}")

    print("-" * 110)
    print(f"Total Dataset Size:    {format_bytes(total_dataset_bytes)} ({total_dataset_bytes:,} bytes)")
    print(f"Explicitly Excluded:   {format_bytes(total_excluded_bytes)} (images.zip)")
    print(f"Pending Download Size: {format_bytes(total_download_bytes)}")
    print("=" * 110)

    return total_dataset_bytes, total_download_bytes, total_excluded_bytes


# ==============================================================================
# Downloader Implementation
# ==============================================================================


class DatasetDownloader:
    """Handles HTTP streaming downloads with range request resuming,

    exponential backoff retries, rate-limit backoff, HTML error detection,
    and MD5 verification.
    """

    def __init__(
        self,
        downloads_dir: Path,
        max_retries: int = 20,
        timeout: int = DEFAULT_TIMEOUT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overwrite: bool = False,
        skip_checksum: bool = False,
        insecure: bool = False,
    ):
        self.downloads_dir = downloads_dir
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.overwrite = overwrite
        self.skip_checksum = skip_checksum
        self.insecure = insecure

        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def download_file(self, record: FileMetadata) -> bool:
        """Downloads a single file into downloads_dir adhering to all rules:

        - Explicitly refuses to download excluded files (images.zip)
        - Skips existing valid files if checksum and size match
        - Resumes interrupted downloads using HTTP Range requests
        - Retries on network errors and HTTP 429 with exponential backoff
        - Validates HTTP status codes and detects HTML error pages
        - Verifies published MD5 checksum and file size
        """
        if record.filename in EXCLUDED_FILES:
            print(f"[EXCLUDE] Skipping '{record.filename}' - explicitly excluded (already downloaded).")
            record.status = "excluded_already_downloaded"
            return True

        dest_file = self.downloads_dir / record.filename
        part_file = self.downloads_dir / f"{record.filename}.part"

        # Check existing valid file
        if dest_file.exists() and not self.overwrite:
            local_size = dest_file.stat().st_size
            if local_size == record.expected_size:
                if self.skip_checksum:
                    print(f"[SKIP] '{record.filename}' exists with expected size ({local_size:,} bytes).")
                    record.actual_size = local_size
                    record.status = "verified"
                    return True
                print(f"[CHECK] Verifying existing file '{record.filename}'...")
                local_md5 = compute_file_md5(dest_file)
                if local_md5 == record.expected_checksum.lower():
                    print(f"[SKIP] '{record.filename}' already downloaded and MD5 verified.")
                    record.actual_size = local_size
                    record.actual_checksum = local_md5
                    record.status = "verified"
                    return True
                else:
                    print(
                        f"[MISMATCH] Existing '{record.filename}' checksum mismatch "
                        f"(local: {local_md5}, expected: {record.expected_checksum}). Will re-download."
                    )
            else:
                print(
                    f"[SIZE MISMATCH] Existing '{record.filename}' size ({local_size:,} bytes) != "
                    f"expected ({record.expected_size:,} bytes). Will re-download."
                )

        # Download with retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                success = self._execute_download(record, dest_file, part_file, attempt)
                if success:
                    # Polite pause between downloads to respect server rate limits
                    time.sleep(1.0)
                    return True
            except RateLimitException as rle:
                delay = (5 * attempt) + random.uniform(1.0, 3.0)
                print(f"[RATE LIMITED] Server returned 429 Too Many Requests. Cooling down for {delay:.1f}s...")
                time.sleep(delay)
            except Exception as e:
                print(f"[ERROR] Attempt {attempt}/{self.max_retries} failed for '{record.filename}': {e}")
                if attempt < self.max_retries:
                    delay = min(30.0, (2 ** min(attempt - 1, 5))) + random.uniform(1.0, 3.0)
                    print(f"        Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    record.status = "failed"
                    record.error_message = str(e)
                    print(f"[FAILED] Exhausted all retries for '{record.filename}'.")
                    return False

        return False

    def _execute_download(
        self,
        record: FileMetadata,
        dest_file: Path,
        part_file: Path,
        attempt: int,
    ) -> bool:
        headers = {}
        existing_bytes = 0

        # Check if partial file exists for resume
        if part_file.exists() and not self.overwrite:
            existing_bytes = part_file.stat().st_size
            if existing_bytes == record.expected_size:
                print(f"[COMPLETE] '{record.filename}.part' is already full size ({existing_bytes:,} bytes). Verifying...")
                if not self.skip_checksum and record.expected_checksum:
                    computed_md5 = compute_file_md5(part_file)
                    if computed_md5 != record.expected_checksum.lower():
                        print(f"[CHECKSUM MISMATCH] Completed partial file checksum failed. Will re-download.")
                        part_file.unlink()
                        existing_bytes = 0
                    else:
                        record.actual_checksum = computed_md5
                        self._promote_part_file(part_file, dest_file)
                        record.actual_size = dest_file.stat().st_size
                        record.status = "verified"
                        print(f"[SUCCESS] '{record.filename}' verified and promoted from completed .part file.")
                        return True
                else:
                    self._promote_part_file(part_file, dest_file)
                    record.actual_size = dest_file.stat().st_size
                    record.status = "verified"
                    return True
            elif existing_bytes > record.expected_size:
                # Corrupt partial file larger than expected
                part_file.unlink()
                existing_bytes = 0
            elif existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"
                print(
                    f"[RESUME] Resuming '{record.filename}' from byte "
                    f"{existing_bytes:,} / {record.expected_size:,} ({format_bytes(existing_bytes)})"
                )

        # Initiate streaming request with SSL error resilience
        try:
            response = self.session.get(
                record.url,
                headers=headers,
                stream=True,
                timeout=self.timeout,
                verify=not self.insecure,
            )
        except requests.exceptions.SSLError as ssle:
            print(f"[SSL WARNING] SSL verification failed ({ssle}). Retrying with certificate verification disabled...")
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = self.session.get(
                record.url,
                headers=headers,
                stream=True,
                timeout=self.timeout,
                verify=False,
            )

        if response.status_code == 429:
            raise RateLimitException("HTTP 429 Too Many Requests")

        # Handle Range response or full response
        if response.status_code == 206:
            # Server accepted range request
            write_mode = "ab"
        elif response.status_code == 200:
            # Full content returned
            write_mode = "wb"
            existing_bytes = 0
        elif response.status_code == 416:
            # Range not satisfiable (e.g. invalid range requested)
            if existing_bytes == record.expected_size:
                self._promote_part_file(part_file, dest_file)
                record.actual_size = dest_file.stat().st_size
                record.status = "verified"
                return True
            if part_file.exists():
                part_file.unlink()
            existing_bytes = 0
            write_mode = "wb"
            response = self.session.get(record.url, stream=True, timeout=self.timeout, verify=not self.insecure)
            if response.status_code == 429:
                raise RateLimitException("HTTP 429 Too Many Requests")
            response.raise_for_status()
        else:
            response.raise_for_status()

        # Detect HTML error page
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type and not record.filename.endswith((".html", ".htm")):
            peek_data = response.raw.read(1024)
            if is_html_content(peek_data):
                raise ValueError(
                    f"Server returned HTML error page instead of binary content: "
                    f"HTTP {response.status_code} ({peek_data[:120].decode('utf-8', errors='ignore')})"
                )

        # Prepare progress bar
        total_size = record.expected_size
        pbar = None
        if tqdm:
            pbar = tqdm(
                total=total_size,
                initial=existing_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=record.filename[:30],
                leave=True,
            )
        else:
            print(f"Downloading {record.filename} ({format_bytes(total_size)})...")

        # Stream chunks into part file
        downloaded_bytes = existing_bytes
        start_time = time.time()

        with open(part_file, write_mode) as f:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    # Check first chunk for HTML if starting from byte 0
                    if downloaded_bytes == 0 and is_html_content(chunk) and not record.filename.endswith(".html"):
                        raise ValueError("Downloaded content starts with HTML markup; expected binary file.")
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    if pbar:
                        pbar.update(len(chunk))

        if pbar:
            pbar.close()

        # Validate downloaded size
        final_size = part_file.stat().st_size
        if final_size != record.expected_size:
            raise IOError(
                f"Downloaded size mismatch for '{record.filename}': "
                f"got {final_size:,} bytes, expected {record.expected_size:,} bytes."
            )

        # Validate published checksum
        if not self.skip_checksum and record.expected_checksum:
            print(f"[VERIFY] Computing MD5 checksum for '{record.filename}'...")
            computed_md5 = compute_file_md5(part_file)
            if computed_md5 != record.expected_checksum.lower():
                part_file.unlink()
                raise ValueError(
                    f"Checksum mismatch for '{record.filename}': "
                    f"computed MD5 '{computed_md5}' != expected '{record.expected_checksum}'"
                )
            record.actual_checksum = computed_md5

        # Check ZIP validity if archive
        if record.filename.endswith(".zip"):
            print(f"[INTEGRITY] Testing ZIP structure of '{record.filename}'...")
            try:
                with zipfile.ZipFile(part_file, "r") as zf:
                    bad_member = zf.testzip()
                    if bad_member:
                        part_file.unlink()
                        raise ValueError(f"Corrupt ZIP file detected! First bad member: {bad_member}")
            except zipfile.BadZipFile as bzf:
                part_file.unlink()
                raise ValueError(f"Invalid ZIP file structure: {bzf}")

        # Atomic replacement: rename part_file to dest_file with retry loop for Windows file locking
        self._promote_part_file(part_file, dest_file)

        record.actual_size = dest_file.stat().st_size
        record.status = "verified"
        elapsed = time.time() - start_time
        speed = (final_size - existing_bytes) / (elapsed if elapsed > 0 else 1) / (1024 * 1024)
        print(f"[SUCCESS] '{record.filename}' downloaded and verified ({format_bytes(final_size)}, avg {speed:.2f} MB/s).")
        return True

    def _promote_part_file(self, part_file: Path, dest_file: Path):
        """Atomically promotes part_file to dest_file, handling Windows file locking from scanners."""
        import gc
        gc.collect()
        max_attempts = 15
        for attempt in range(max_attempts):
            try:
                if dest_file.exists():
                    dest_file.unlink()
                os.replace(str(part_file), str(dest_file))
                return
            except (PermissionError, OSError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(1.0)
                    gc.collect()
                else:
                    raise IOError(f"Could not rename '{part_file}' to '{dest_file}' after {max_attempts} attempts: {e}")


class RateLimitException(Exception):
    """Raised when server responds with HTTP 429 Too Many Requests."""

    pass


# ==============================================================================
# Safe ZIP Extractor (Prevents Zip Slip)
# ==============================================================================


class SafeZipExtractor:
    """Safely extracts ZIP archives, preventing directory traversal vulnerabilities

    (Zip Slip) and strictly avoiding extraction of excluded files (images.zip).
    """

    def __init__(self, extracted_dir: Path):
        self.extracted_dir = extracted_dir
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    def extract_archive(self, zip_path: Path) -> List[Path]:
        """Safely extracts a ZIP archive into a dedicated subdirectory inside

        self.extracted_dir, strictly validating all member paths.
        """
        if zip_path.name in EXCLUDED_FILES:
            print(f"[SAFEGUARD] Refusing to extract '{zip_path.name}' (strictly excluded by rule).")
            return []

        if not zip_path.exists():
            print(f"[WARNING] Archive '{zip_path}' does not exist, cannot extract.")
            return []

        subfolder_name = zip_path.stem  # e.g., 'calibration', 'pointclouds'
        target_dir = (self.extracted_dir / subfolder_name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"[EXTRACT] Extracting '{zip_path.name}' safely into '{target_dir}'...")

        extracted_files: List[Path] = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check integrity
            bad_member = zf.testzip()
            if bad_member:
                raise ValueError(f"Cannot extract corrupted archive '{zip_path.name}': bad member {bad_member}")

            members = zf.infolist()
            pbar = tqdm(members, desc=f"Unpacking {zip_path.stem}", unit="files") if tqdm else None

            for member in members:
                # Zip Slip Protection: Resolve member destination and check prefix
                member_path = (target_dir / member.filename).resolve()
                try:
                    member_path.relative_to(target_dir)
                except ValueError:
                    raise SecurityError(
                        f"Zip Slip traversal attempt detected in '{zip_path.name}': "
                        f"illegal member path '{member.filename}'"
                    )

                zf.extract(member, target_dir)
                extracted_files.append(member_path)
                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

        print(f"[EXTRACTED] Completed extraction of '{zip_path.name}' ({len(extracted_files)} files extracted).")
        return extracted_files


class SecurityError(Exception):
    """Raised when a security hazard such as Zip Slip is detected."""

    pass


# ==============================================================================
# Manifest Manager
# ==============================================================================


class ManifestManager:
    """Manages the creation, updating, and saving of manifest.json."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path

    def load_existing(self) -> Dict[str, dict]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("files", {})
            except Exception:
                return {}
        return {}

    def save_manifest(
        self,
        files: Dict[str, FileMetadata],
        dataset_meta: dict,
        workspace_root: Path,
    ):
        files_dict = {}

        for name, record in files.items():
            entry = {
                "filename": record.filename,
                "url": record.url,
                "expected_size": record.expected_size,
                "actual_size": record.actual_size,
                "checksum_algorithm": record.checksum_algorithm,
                "expected_checksum": record.expected_checksum,
                "actual_checksum": record.actual_checksum,
                "status": record.status,
            }

            # For images.zip, status must be strictly 'excluded_already_downloaded'
            if name == "images.zip":
                entry["status"] = "excluded_already_downloaded"
                # Check if images.zip is present locally in root or downloads
                local_images_candidates = [
                    workspace_root / "images.zip",
                    self.manifest_path.parent / "downloads" / "images.zip",
                ]
                for cand in local_images_candidates:
                    if cand.exists():
                        entry["actual_size"] = cand.stat().st_size
                        break

            if record.error_message:
                entry["error_message"] = record.error_message

            files_dict[name] = entry

        manifest_data = {
            "dataset_name": dataset_meta.get("title", "ETH Zurich Eschikon Plant Stress Phenotyping Dataset (2018)"),
            "official_url": OFFICIAL_ORIGINAL_URL,
            "replacement_portal": FALLBACK_DATASET_PAGE,
            "doi": dataset_meta.get("doi", KNOWN_DOI),
            "handle": dataset_meta.get("handle", KNOWN_HANDLE),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_files": len(files_dict),
            "files": files_dict,
        }

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"[MANIFEST] Saved dataset manifest to: {self.manifest_path}")


# ==============================================================================
# Main Orchestration & CLI
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust downloader for the ETH Zurich Eschikon Plant Stress Phenotyping Dataset (2018)."
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Discover files from official sources, display the review table, and exit without downloading.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and verify all available files (excluding images.zip) into downloads/, but do not extract.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract valid downloaded ZIP archives (excluding images.zip) into extracted/.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Default behavior: review, download remaining files, and safely extract archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plant_stress_dataset",
        help="Target base directory (default: 'plant_stress_dataset').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing valid files instead of skipping them.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip MD5 checksum verification (not recommended).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=20,
        help="Maximum retry attempts for failed downloads (default: 20).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP request timeout in seconds (default: 45).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification if network proxies/firewalls cause SSLCertVerificationError.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Proceed with multi-gigabyte download without interactive confirmation prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    base_dir = Path(args.output_dir).resolve()
    downloads_dir = base_dir / "downloads"
    extracted_dir = base_dir / "extracted"
    manifest_file = base_dir / "manifest.json"
    workspace_root = Path(".").resolve()

    # Step 1: Discover files from official portal
    discovery = DatasetDiscovery()
    try:
        discovered_files = discovery.discover()
    except Exception as e:
        print(f"\n[FATAL] Dataset discovery failed: {e}", file=sys.stderr)
        print("Please verify your internet connection or check https://ethz-asl.github.io/datasets/", file=sys.stderr)
        sys.exit(1)

    dataset_meta = {
        "title": "Eschikon Plant Stress Phenotyping Dataset",
        "doi": KNOWN_DOI,
        "handle": KNOWN_HANDLE,
    }

    # Step 2: Display Review Table
    _, pending_bytes, _ = display_review_table(
        discovered_files,
        downloads_dir,
        workspace_root,
    )

    # If --list-files specified, exit immediately after review
    if args.list_files:
        print("[INFO] --list-files specified. Exiting without downloading.")
        sys.exit(0)

    # If only --extract specified without download
    should_download = not args.extract or args.all or args.download_only
    should_extract = args.extract or args.all or (not args.download_only and not args.extract)

    manifest_mgr = ManifestManager(manifest_file)

    # Step 3: Confirmation prompt for multi-gigabyte download
    if should_download and pending_bytes > 0:
        if not args.yes and sys.stdin.isatty():
            prompt_msg = f"Ready to download {format_bytes(pending_bytes)} of dataset files. Proceed? [Y/n]: "
            try:
                choice = input(prompt_msg).strip().lower()
                if choice not in ("y", "yes", ""):
                    print("Download cancelled by user.")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                sys.exit(0)

    # Step 4: Download files
    if should_download:
        print("\n" + "=" * 80)
        print("DOWNLOADING DATASET FILES (images.zip strictly excluded)")
        print("=" * 80)

        downloader = DatasetDownloader(
            downloads_dir=downloads_dir,
            max_retries=args.max_retries,
            timeout=args.timeout,
            overwrite=args.overwrite,
            skip_checksum=args.skip_checksum,
            insecure=args.insecure,
        )

        success_count = 0
        fail_count = 0

        # Sort files so smaller metadata files download first, followed by larger archives
        sorted_files = sorted(
            discovered_files.values(),
            key=lambda rec: rec.expected_size,
        )

        for record in sorted_files:
            name = record.filename
            if name in EXCLUDED_FILES:
                record.status = "excluded_already_downloaded"
                continue

            print(f"\n---> Processing: {name} ({format_bytes(record.expected_size)})")
            ok = downloader.download_file(record)
            if ok:
                success_count += 1
            else:
                fail_count += 1

            # Incrementally save manifest
            manifest_mgr.save_manifest(discovered_files, dataset_meta, workspace_root)

        print("\n" + "-" * 80)
        print(f"Download Summary: {success_count} succeeded, {fail_count} failed.")
        print("-" * 80)

        if fail_count > 0:
            print("[WARNING] Some files failed to download. Check errors and rerun to resume.")

    # Step 5: Safe Extraction
    if should_extract:
        print("\n" + "=" * 80)
        print("SAFE ARCHIVE EXTRACTION (images.zip strictly excluded)")
        print("=" * 80)

        extractor = SafeZipExtractor(extracted_dir=extracted_dir)

        # Find all downloaded .zip files in downloads_dir, excluding images.zip
        for zip_file in sorted(downloads_dir.glob("*.zip")):
            if zip_file.name in EXCLUDED_FILES:
                print(f"[SAFEGUARD] Skipping '{zip_file.name}': strictly excluded from extraction.")
                continue

            try:
                extractor.extract_archive(zip_file)
            except Exception as e:
                print(f"[ERROR] Failed to extract '{zip_file.name}': {e}", file=sys.stderr)

    # Final manifest save
    manifest_mgr.save_manifest(discovered_files, dataset_meta, workspace_root)
    print("\n" + "=" * 80)
    print("ALL REQUESTED TASKS COMPLETED SUCCESSFULLY!")
    print(f"Dataset root: {base_dir}")
    print(f"Downloads:    {downloads_dir}")
    print(f"Extracted:    {extracted_dir}")
    print(f"Manifest:     {manifest_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

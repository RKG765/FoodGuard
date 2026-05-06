"""
Corrupted Image Scanner & Cleaner
===================================
Scans all AI-generated image folders and detects:
  - 0-byte files
  - Files too small to be real images (< 1KB)
  - Truncated / unreadable images (PIL can't decode)
  - Images with broken dimensions (< 32px)
  - RGBA / palette mode images that need conversion

Run:  python scripts/cleanup_corrupted.py
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageFile
import warnings

# Allow PIL to attempt loading truncated images for detection
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)
Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path("e:/BML/Semester-VI/Prj-3")
AI_DIR = PROJECT_ROOT / "ai_generated"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_FILE_SIZE = 1024   # 1 KB minimum
MIN_DIM = 32           # minimum width/height in pixels


def scan_folder(folder_path):
    """Scan a single folder for bad images. Returns dict of issues."""
    issues = {
        "zero_byte": [],
        "too_small_file": [],
        "corrupted": [],
        "bad_dimensions": [],
        "wrong_mode": [],
    }
    good_count = 0

    for img_path in sorted(folder_path.rglob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        file_size = img_path.stat().st_size

        # Check 1: Zero bytes
        if file_size == 0:
            issues["zero_byte"].append(img_path)
            continue

        # Check 2: Tiny file
        if file_size < MIN_FILE_SIZE:
            issues["too_small_file"].append((img_path, file_size))
            continue

        # Check 3: Can we open and decode it?
        try:
            with Image.open(img_path) as img:
                img.verify()

            # Re-open to actually load pixels (catches truncated data)
            with Image.open(img_path) as img:
                img.load()
                w, h = img.size
                mode = img.mode

                # Check 4: Absurd dimensions
                if w < MIN_DIM or h < MIN_DIM:
                    issues["bad_dimensions"].append((img_path, w, h))
                    continue

                # Check 5: Non-RGB mode (not fatal, just flag)
                if mode not in ("RGB",):
                    issues["wrong_mode"].append((img_path, mode))

                good_count += 1

        except Exception as e:
            issues["corrupted"].append((img_path, str(e)[:80]))

    return issues, good_count


def print_issues(folder_name, issues, good_count):
    """Pretty-print issues for a folder."""
    total_bad = (
        len(issues["zero_byte"])
        + len(issues["too_small_file"])
        + len(issues["corrupted"])
        + len(issues["bad_dimensions"])
    )

    status = "✅" if total_bad == 0 else "⚠️"
    print(f"\n{status} {folder_name}/ — {good_count} good, {total_bad} bad")

    if issues["zero_byte"]:
        print(f"   ❌ Zero-byte files: {len(issues['zero_byte'])}")
        for p in issues["zero_byte"][:5]:
            print(f"      {p.name}")

    if issues["too_small_file"]:
        print(f"   ❌ Too small (< 1KB): {len(issues['too_small_file'])}")
        for p, sz in issues["too_small_file"][:5]:
            print(f"      {p.name} ({sz} bytes)")

    if issues["corrupted"]:
        print(f"   ❌ Corrupted/unreadable: {len(issues['corrupted'])}")
        for p, err in issues["corrupted"][:5]:
            print(f"      {p.name} — {err}")

    if issues["bad_dimensions"]:
        print(f"   ❌ Bad dimensions (< 32px): {len(issues['bad_dimensions'])}")
        for p, w, h in issues["bad_dimensions"][:5]:
            print(f"      {p.name} ({w}x{h})")

    if issues["wrong_mode"]:
        print(f"   ⚡ Non-RGB mode (auto-fixable): {len(issues['wrong_mode'])}")

    return total_bad


def collect_deletable(issues):
    """Collect all paths that should be deleted."""
    paths = []
    paths.extend(issues["zero_byte"])
    paths.extend([p for p, _ in issues["too_small_file"]])
    paths.extend([p for p, _ in issues["corrupted"]])
    paths.extend([p for p, _, _ in issues["bad_dimensions"]])
    return paths


def fix_wrong_mode(issues):
    """Convert non-RGB images to RGB in-place."""
    fixed = 0
    for img_path, mode in issues["wrong_mode"]:
        try:
            with Image.open(img_path) as img:
                rgb = img.convert("RGB")
                rgb.save(img_path)
                fixed += 1
        except Exception:
            pass
    return fixed


def main():
    print("=" * 60)
    print("  FOODGUARD — CORRUPTED IMAGE SCANNER")
    print("=" * 60)
    print(f"  Scanning: {AI_DIR}")
    print(f"  Min file size: {MIN_FILE_SIZE} bytes")
    print(f"  Min dimensions: {MIN_DIM}x{MIN_DIM} px")

    # Get all subfolders
    folders = sorted([d for d in AI_DIR.iterdir() if d.is_dir()])

    all_deletable = []
    all_fixable_modes = []
    total_good = 0
    total_bad = 0

    for folder in folders:
        print(f"\n  Scanning {folder.name}/...", end="", flush=True)
        issues, good_count = scan_folder(folder)
        bad_count = print_issues(folder.name, issues, good_count)

        all_deletable.extend(collect_deletable(issues))
        all_fixable_modes.extend(issues["wrong_mode"])
        total_good += good_count
        total_bad += bad_count

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total good images:    {total_good}")
    print(f"  Total bad images:     {total_bad}")
    print(f"  Non-RGB (fixable):    {len(all_fixable_modes)}")

    # Step A: Fix non-RGB
    if all_fixable_modes:
        print(f"\n  Converting {len(all_fixable_modes)} non-RGB images to RGB...")
        fixed = fix_wrong_mode({"wrong_mode": all_fixable_modes})
        print(f"  ✅ Fixed {fixed} images to RGB mode.")

    # Step B: Delete corrupted
    if all_deletable:
        print(f"\n  🗑️  {len(all_deletable)} corrupted images found.")
        print(f"  Files to delete:")
        for p in all_deletable[:30]:
            print(f"    {p.relative_to(PROJECT_ROOT)}")
        if len(all_deletable) > 30:
            print(f"    ... and {len(all_deletable) - 30} more")

        print()
        response = input("  Delete all corrupted files? (yes/no): ").strip().lower()
        if response == "yes":
            deleted = 0
            for p in all_deletable:
                try:
                    p.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"    Failed: {p.name} — {e}")
            print(f"\n  ✅ Deleted {deleted} corrupted files.")
        else:
            print("  ❌ Aborted. No files deleted.")
    else:
        print(f"\n  ✅ All images are clean! Nothing to delete.")

    # Final counts
    print("\n" + "=" * 60)
    print("  FINAL IMAGE COUNTS (after cleanup)")
    print("=" * 60)
    for folder in folders:
        imgs = [f for f in folder.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"  {folder.name:25s}: {len(imgs):5d} images")


if __name__ == "__main__":
    main()

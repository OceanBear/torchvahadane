#!/usr/bin/env python3
"""
Convert PNG/JPG/JPEG images in a folder to TIFF (same directory).

Default folder: /mnt/d/BCCRC-work/TNBC_NucleiSegmentation/tiles_scn
"""

import argparse
from pathlib import Path

from PIL import Image

DEFAULT_DIR = Path("/mnt/d/BCCRC-work/TNBC_NucleiSegmentation/tiles_scn")
SOURCE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def convert_image_to_tiff(src: Path, remove_source: bool = False) -> Path:
    """Convert one image to TIFF; return output path."""
    dst = src.with_suffix(".tiff")
    if dst.exists():
        raise FileExistsError(f"Output already exists: {dst.name}")

    with Image.open(src) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(dst, format="TIFF")

    if remove_source:
        src.unlink()

    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PNG/JPG/JPEG images in a folder to TIFF (same folder)."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Folder containing images (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete the original PNG/JPG/JPEG after a successful conversion.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip conversion when the .tiff already exists (default: on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Overwrite or recreate TIFF even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.dir

    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    sources = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS
    )
    if not sources:
        print(f"No PNG/JPG/JPEG files found in {folder}")
        return

    converted = 0
    skipped = 0
    failed = 0

    for src in sources:
        dst = src.with_suffix(".tiff")
        if args.skip_existing and dst.exists():
            print(f"  Skip (exists): {dst.name}")
            skipped += 1
            continue

        if dst.exists() and not args.skip_existing:
            dst.unlink()

        try:
            out = convert_image_to_tiff(src, remove_source=args.remove_source)
            print(f"  {src.name} -> {out.name}")
            converted += 1
        except Exception as exc:
            print(f"  Failed: {src.name} ({exc})")
            failed += 1

    print(
        f"Done. converted={converted}, skipped={skipped}, failed={failed}, folder={folder}"
    )


if __name__ == "__main__":
    main()

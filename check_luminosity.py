"""
Check mean luminosity of each TIFF tile in the folder.
Uses LAB L (lightness), normalized to 0–1.
Downsamples large images for memory efficiency.
"""

from pathlib import Path
import sys

try:
    from PIL import Image
except ImportError:
    print("Please install Pillow: pip install Pillow")
    sys.exit(1)

# Input directory containing TIFF tiles (WSL path for
# J:\HandE\results\SOW1885_n=201_AT2 40X\JN_TS_test\original_tiles)
# /mnt/j/HandE/results/SOW1885_n=201_AT2 40X/Batch_105/pred/tiles_manual
INPUT_DIR = Path("/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_001-013/pred/std_output")

# D65 reference white for XYZ
_Xn, _Yn, _Zn = 0.95047, 1.0, 1.08883


def _srgb_to_linear(c: float) -> float:
    """sRGB component [0,1] to linear."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_lab_l(r: int, g: int, b: int) -> float:
    """Convert RGB (0–255) to LAB L in range 0–1."""
    rl = _srgb_to_linear(r / 255.0)
    gl = _srgb_to_linear(g / 255.0)
    bl = _srgb_to_linear(b / 255.0)
    # sRGB D65 -> XYZ
    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
    # XYZ -> LAB (L in 0–100)
    def f(t: float) -> float:
        delta = 6 / 29
        if t > delta**3:
            return t ** (1 / 3)
        return t / (3 * delta**2) + 4 / 29

    l = 116 * f(y / _Yn) - 16
    # Clamp L to [0, 100] then normalize to [0, 1]
    l = max(0.0, min(100.0, l))
    return l / 100.0


def get_mean_luminosity(path: Path, max_dim: int = 512) -> float | None:
    """Calculate mean LAB L (0–1) of an image. Downsamples if larger than max_dim."""
    try:
        with Image.open(path) as img:
            # Handle multi-page TIFFs (use first page)
            if hasattr(img, "n_frames") and img.n_frames > 1:
                img.seek(0)
            # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Downsample for memory efficiency
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    total = sum(_rgb_to_lab_l(r, g, b) for r, g, b in pixels)
    return total / len(pixels)


def main():
    folder = INPUT_DIR
    tiffs = sorted(folder.glob("*.tiff")) + sorted(folder.glob("*.tif"))

    if not tiffs:
        print("No TIFF files found.")
        return

    print(f"{'Filename':<50} {'Luminosity':>10}")
    print("-" * 62)

    results = []
    for path in tiffs:
        lum = get_mean_luminosity(path)
        if lum is not None:
            results.append((path.name, lum))
            print(f"{path.name:<50} {lum:>10.4f}")

    if results:
        avg = sum(l for _, l in results) / len(results)
        print("-" * 62)
        print(f"{'Average':<50} {avg:>10.4f}")


if __name__ == "__main__":
    main()

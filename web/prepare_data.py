from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH_JSON = ROOT / "output" / "flower_path.json"
OUTPUT_JS = Path(__file__).resolve().parent / "flower-data.js"

IMAGE_SRC = ROOT / "output" / "flower_recolored.png"
IMAGE_DEST = Path(__file__).resolve().parent / "flower.png"


def transform_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center_x = points[:, 1].max() / 2
    center_y = points[:, 0].max() / 2
    center = np.array([center_x, center_y], dtype=float)
    shifted = points - center

    max_size = np.linalg.norm(shifted, axis=1).max()
    scaled = shifted / max_size
    scaled[:, 0] *= -1
    return scaled, center, float(max_size)


def main() -> None:
    shutil.copy(IMAGE_SRC, IMAGE_DEST)

    points = np.array(json.loads(PATH_JSON.read_text(encoding="utf-8")), dtype=float)
    transformed, center, max_size = transform_points(points)

    z = transformed[:, 1] + 1j * transformed[:, 0]
    n = len(z)
    draw_scale = 3.2
    compute_vectors = 1000
    render_vectors = 1000

    coeffs = np.fft.fft(z) / n * draw_scale
    freqs = np.fft.fftfreq(n, d=1 / n)
    sorted_indices = np.argsort(np.abs(freqs))
    coeffs = coeffs[sorted_indices][:compute_vectors]
    freqs = freqs[sorted_indices][:compute_vectors]

    data = {
        "drawScale": draw_scale,
        "computeVectors": compute_vectors,
        "renderVectors": render_vectors,
        "center": center.tolist(),
        "maxSize": max_size,
        "coefficients": [
            [float(freq), float(coeff.real), float(coeff.imag)]
            for freq, coeff in zip(freqs, coeffs, strict=True)
        ],
    }

    OUTPUT_JS.write_text(
        "window.FLOWER_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

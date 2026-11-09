from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


SurfaceFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SimulationResult:
    reference: np.ndarray
    observed: np.ndarray
    displacement: np.ndarray
    topography: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def save_simulation(
        self,
        directory: str | Path,
        prefix: str = "",
    ) -> dict[str, str]:
        """Save the simulation arrays and rendered images to disk."""
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_prefix = f"{prefix}_" if prefix else ""
        saved_paths: dict[str, str] = {}

        archive_path = output_dir / f"{file_prefix}simulation.npz"
        np.savez_compressed(
            archive_path,
            reference=self.reference,
            observed=self.observed,
            displacement=self.displacement,
            topography=self.topography,
            x=self.x,
            y=self.y,
        )
        saved_paths["archive"] = str(archive_path)

        def save_image(name: str, image: np.ndarray, *, cmap: str = "gray", vmin=None, vmax=None) -> None:
            image_path = output_dir / f"{file_prefix}{name}.png"
            plt.imsave(image_path, image, cmap=cmap, vmin=vmin, vmax=vmax)
            saved_paths[name] = str(image_path)

        save_image("reference", self.reference, cmap="gray", vmin=0.0, vmax=1.0)
        save_image("observed", self.observed, cmap="gray", vmin=0.0, vmax=1.0)
        save_image("topography", self.topography, cmap="gray")
        displacement_magnitude = np.hypot(
            self.displacement[..., 0], self.displacement[..., 1]
        )
        save_image("displacement_magnitude", displacement_magnitude, cmap="magma")

        if self.displacement.shape[-1] >= 2:
            save_image("displacement_x", self.displacement[..., 0], cmap="viridis")
            save_image("displacement_y", self.displacement[..., 1], cmap="viridis")

        return saved_paths


def make_checkerboard(size: int = 512, squares: int = 8) -> np.ndarray:
    """Return a square grayscale checkerboard with values 0 (black) and 1 (white)."""
    if size < 1:
        raise ValueError("size must be positive")
    if squares < 1:
        raise ValueError("squares must be positive")

    rows, columns = np.indices((size, size))
    row_squares = rows * squares // size
    column_squares = columns * squares // size
    return ((row_squares + column_squares) % 2).astype(np.float32)


def capillary_wave(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float = 0.02,
    wavelength: float = 0.18,
    decay_length: float = 0.35,
) -> np.ndarray:
    """A radially symmetric, exponentially damped ripple centered in the bath."""
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if decay_length <= 0:
        raise ValueError("decay_length must be positive")

    radius = np.hypot(x, y)
    return amplitude * np.exp(-radius / decay_length) * np.cos(
        2 * np.pi * radius / wavelength
    )


def _sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample an image at floating-point pixel coordinates, clamping at its edges."""
    height, width = image.shape
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0

    return (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    ).astype(np.float32)


def simulate_checkerboard(
    size: int = 512,
    squares: int = 128,
    topography: SurfaceFunction | None = None,
    wave_amplitude: float = 0.02,
    wave_wavelength: float = 0.18,
    wave_decay_length: float = 0.35,
    optical_gain: float = 5.0,
) -> SimulationResult:
    """Simulate a top-down checkerboard view distorted by a fluid surface.

    The first-order optical model maps the surface gradient to a pixel displacement;
    ``optical_gain`` absorbs fluid depth, refractive-index contrast, and pixel scale.
    Displacements are stored as ``(u_x, u_y)`` in pixels, with x right and y down,
    following ``observed(r) = reference(r - displacement(r))``.
    Coordinates passed to ``topography`` span [-0.5, 0.5] across the bath.
    """
    if size < 2:
        raise ValueError("size must be at least 2")

    coordinates = np.linspace(-0.5, 0.5, size, dtype=np.float32)
    x, y = np.meshgrid(coordinates, coordinates)
    surface_function = topography or (
        lambda grid_x, grid_y: capillary_wave(
            grid_x,
            grid_y,
            amplitude=wave_amplitude,
            wavelength=wave_wavelength,
            decay_length=wave_decay_length,
        )
    )
    surface = np.asarray(surface_function(x, y), dtype=np.float32)
    if surface.shape != (size, size):
        raise ValueError("topography function must return an array matching the canvas")

    derivative_y, derivative_x = np.gradient(surface, coordinates, coordinates)
    displacement = np.stack(
        (-optical_gain * derivative_x, -optical_gain * derivative_y), axis=-1
    ).astype(np.float32)

    rows, columns = np.indices((size, size), dtype=np.float32)
    observed = _sample_bilinear(
        make_checkerboard(size=size, squares=squares),
        columns - displacement[..., 0],
        rows - displacement[..., 1],
    )
    return SimulationResult(
        reference=make_checkerboard(size=size, squares=squares),
        observed=observed,
        displacement=displacement,
        topography=surface,
        x=x,
        y=y,
    )


def render_simulation(result: SimulationResult) -> tuple[Figure, np.ndarray]:
    """Render the reference, distorted image, and displacement magnitude."""
    displacement_magnitude = np.hypot(
        result.displacement[..., 0], result.displacement[..., 1]
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(result.reference, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Reference checkerboard")
    axes[1].imshow(result.observed, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    axes[1].set_title("Simulated camera image")
    image = axes[2].imshow(displacement_magnitude, cmap="magma", origin="upper")
    axes[2].set_title("Displacement magnitude (pixels)")
    figure.colorbar(image, ax=axes[2], shrink=0.8)
    for axis in axes:
        axis.set_axis_off()
    return figure, axes

if __name__ == "__main__":
    simulation = simulate_checkerboard()
    render_simulation(simulation)
    simulation.save_simulation('.out/test1')
    plt.show()
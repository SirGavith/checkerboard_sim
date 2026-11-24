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
    height_field: np.ndarray
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
        (output_dir / 'observed').mkdir(parents=True, exist_ok=True)

        file_prefix = f"{prefix}_" if prefix else ""
        saved_paths: dict[str, str] = {}

        archive_path = output_dir / f"{file_prefix}simulation.npz"
        np.savez_compressed(
            archive_path,
            reference=self.reference,
            observed=self.observed,
            displacement=self.displacement,
            topography=self.height_field,
            x=self.x,
            y=self.y,
        )
        saved_paths["archive"] = str(archive_path)

        def save_image(name: str, image: np.ndarray, *, cmap: str = "gray", vmin=None, vmax=None) -> None:
            image_path = output_dir / f"{file_prefix}{name}.png"
            plt.imsave(image_path, image, cmap=cmap, vmin=vmin, vmax=vmax)
            saved_paths[name] = str(image_path)

        save_image("reference", self.reference, cmap="gray", vmin=0.0, vmax=1.0)
        save_image("observed/observed", self.observed, cmap="gray", vmin=0.0, vmax=1.0)
        save_image("topography", self.height_field, cmap="gray")
        displacement_magnitude = np.hypot(
            self.displacement[..., 0], self.displacement[..., 1]
        )
        save_image("displacement_magnitude", displacement_magnitude, cmap="magma")

        if self.displacement.shape[-1] >= 2:
            save_image("displacement_x", self.displacement[..., 0], cmap="viridis")
            save_image("displacement_y", self.displacement[..., 1], cmap="viridis")

        return saved_paths


def make_checkerboard(size: int = 512, squares: int = 8) -> np.ndarray:
    """Return the two-wave grayscale pattern with the former checkerboard density."""
    if size < 1:
        raise ValueError("size must be positive")
    if squares < 1:
        raise ValueError("squares must be positive")

    rows, columns = np.indices((size, size), dtype=np.float32)
    wave_number = np.pi * squares / size
    return (
        0.5 + 0.25 * (np.cos(wave_number * columns) + np.cos(wave_number * rows))
    ).astype(np.float32)


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
    size: int,
    squares: int,
    fluid_depth: float,
    fluid_ior: float ,
    acrylic_thickness: float,
    acrylic_ior: float ,
    topography: SurfaceFunction | None = None,
    wave_amplitude: float = 0.02,
    wave_wavelength: float = 0.18,
    wave_decay_length: float = 0.35,
) -> SimulationResult:
    """Simulate a periodic wave pattern viewed through a fluid surface.

    The first-order optical model maps the surface gradient to a pixel displacement;
    the optical sensitivity is derived from the pixel scale and effective optical
    depth. The normalized bath coordinates span one drop diameter.
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

    displac = compute_displacement(surface, coordinates,
                                   fluid_depth, fluid_ior, acrylic_thickness, acrylic_ior)
    displacement = np.stack(
        (-displac[0], -displac[1]),
        axis=-1,
    ).astype(np.float32)

    rows, columns = np.indices((size, size), dtype=np.float32)
    reference = make_checkerboard(size=size, squares=squares)
    observed = _sample_bilinear(
        reference,
        columns - displacement[..., 0],
        rows - displacement[..., 1],
    )
    return SimulationResult(
        reference=reference,
        observed=observed,
        displacement=displacement,
        height_field=surface,
        x=x,
        y=y,
    )

def compute_displacement(surface, coordinates, fluid_depth, fluid_ior, acrylic_thickness, acrylic_ior, n_a=1.0):
    """
    Computes the exact non-paraxial horizontal displacement field (u_x, u_y)
    of a bottom pattern viewed from above through a wavy fluid surface.
    
    Parameters:
        surface: 2D numpy array of surface elevations eta(y, x)
        coordinates: 1D array or scalar grid spacing
        fluid_depth: Depth of the fluid layer
        fluid_ior: Refractive index of the fluid
        acrylic_thickness: Thickness of the acrylic layer
        acrylic_ior: Refractive index of the acrylic
        n_a: Refractive index of the ambient medium
    Returns:
        u_x, u_y: 2D numpy arrays of apparent horizontal displacement
    """
    derivative_y, derivative_x = np.gradient(surface, coordinates, coordinates)
    
    # Squared gradient magnitude G^2 = (d_eta/dx)^2 + (d_eta/dy)^2
    G_sq = derivative_x**2 + derivative_y**2
    
    # Ratio of refractive indices
    n_al = n_a / fluid_ior
    n_lc = fluid_ior / acrylic_ior
    
    # Exact factor: (sin(theta_l) / G), well-behaved even when G == 0
    sin_theta_l_over_G = (
        np.sqrt(1.0 + (1.0 - n_al**2) * G_sq) - n_al
    ) / (1.0 + G_sq)
    
    # sin^2(theta_l) and sin^2(theta_c)
    sin_sq_theta_l = (sin_theta_l_over_G**2) * G_sq
    sin_sq_theta_c = (n_lc**2) * sin_sq_theta_l
    
    # Local fluid depth (constant h_l, or h_l + eta if finite amplitude is included)
    h_l = (fluid_depth + surface)
    
    # Effective non-paraxial depth factor h*_exact(x, y)
    h_star_exact = sin_theta_l_over_G * (
        h_l / np.sqrt(1.0 - sin_sq_theta_l)
        + (n_lc * acrylic_thickness) / np.sqrt(1.0 - sin_sq_theta_c)
    )

    # print(f"Max h_star_exact: {np.max(h_star_exact)}, Min h_star_exact: {np.min(h_star_exact)}")
    # print(f"Average h_star_exact: {np.mean(h_star_exact)}")

    u_x = h_star_exact * derivative_x
    u_y = h_star_exact * derivative_y
    
    return u_x, u_y

def render_simulation(result: SimulationResult) -> tuple[Figure, np.ndarray]:
    """Render the reference and distorted image."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(result.reference, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Reference pattern")
    axes[1].imshow(result.observed, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    axes[1].set_title("Simulated camera image")
    for axis in axes:
        axis.set_axis_off()
    return figure, axes
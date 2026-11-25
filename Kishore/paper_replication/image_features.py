"""Measurements on explicit CyMorph-like preprocessing intermediates.

Preprocessing and catalog production equivalence require separate validation.
The adapters here never accept rendered RGB.

Numerical conventions adapted from the GPL-3.0 CyMorph authors' source at
1ed0ee4c3de7ad1282aa24d5ada0666229e59d7f. See LICENSE-CyMorph.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft, ndimage
from scipy.stats import spearmanr

from common import FEATURES


@dataclass(frozen=True)
class Settings:
    inner_fraction: float = .35
    outer_fraction: float = .75
    entropy_bins: int = 130
    butterworth_order: int = 2
    butterworth_degree: float = .2
    module_tolerance: float = .02
    phase_tolerance_rad: float = .03
    radial_tolerance_pixel: float = .01


def concentration(acc, rp, inner=.35, outer=.75):
    """Use the 2 Rp aperture and the public code's legacy interpolation."""
    edge = 2 * rp
    if not np.isfinite(edge) or edge <= 0 or np.ceil(edge) >= len(acc):
        return np.nan
    low, high = int(np.floor(edge)), int(np.ceil(edge))
    total = acc[low] + (acc[high] - acc[low]) * (edge - np.floor(rp))
    if total <= 0:
        return np.nan

    def crossing(fraction):
        threshold = fraction * total
        found = np.flatnonzero(acc[1:] > threshold)
        if not len(found):
            return np.nan
        i = found[0]
        return i + (threshold - acc[i]) / (acc[i+1] - acc[i]) if acc[i+1] > acc[i] else np.nan

    r1, r2 = crossing(inner), crossing(outer)
    return float(np.log10(r2/r1)) if r1 > 0 and r2 >= r1 else np.nan


def rank_distance(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(1 - spearmanr(a, b).statistic)


def asymmetry(image, mask, fwhm):
    """Rotate both input and mask 180 degrees about the array center."""
    rotated = ndimage.rotate(image, 180, reshape=False, mode="nearest", order=3)
    rotated_mask = ndimage.rotate((~mask).astype(float), 180, reshape=False, mode="nearest", order=3)
    y, x = np.indices(image.shape)
    height, width = image.shape
    # Preserve the public implementation's far-corner coordinate convention.
    support = mask & (rotated_mask <= .5) & (np.hypot(x-width, y-height) > .2*fwhm)
    return rank_distance(image[support], rotated[support])


def butterworth(image, max_rad, settings):
    height, width = image.shape
    y, x = np.indices(image.shape)
    distance = np.hypot(y-height/2, x-width/2)
    response = 1 / (1 + (distance/(settings.butterworth_degree*max_rad))**(2*settings.butterworth_order))
    return fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(image))*response)).real


def smoothness(image, filter_input, mask, max_rad, fwhm, settings):
    smoothed = butterworth(filter_input, max_rad, settings)
    y, x = np.indices(image.shape)
    height, width = image.shape
    support = mask & (np.hypot(x-width//2, y-height//2) > .2*fwhm)
    return rank_distance(image[support], smoothed[support]) if np.count_nonzero(support) >= 6 else np.nan


def entropy(image, mask, bins):
    pixels = image[mask & (image != 0)]
    if not len(pixels):
        return np.nan
    counts, _ = np.histogram(pixels, bins=bins)
    p = counts[counts > 0]/len(pixels)
    return float(-np.sum(p*np.log(p))/np.log(bins))


def gradient_moment(vectors, population):
    """Rosa et al. second gradient moment from surviving vectors."""
    if population == 0:
        return np.nan
    if len(vectors) == 0:
        return 0.
    alignment = np.linalg.norm(vectors.sum(axis=0))/np.linalg.norm(vectors, axis=1).sum()
    return float(len(vectors)/population*(2-alignment))


def g2(image, mask, cx, cy, settings):
    """Translate the released GPA ring and mask conventions to NumPy."""
    dy, dx = np.gradient(np.asarray(image, dtype=np.float32))
    magnitude = np.hypot(dx, dy)
    peak = float(magnitude.max())
    if peak == 0 or not mask.any():
        return np.nan
    phase = np.arctan2(dy, dx)
    y, x = np.indices(image.shape)
    rings = np.hypot(x-cx, y-cy).astype(int)
    threshold = settings.module_tolerance * peak
    keep = mask & (magnitude > threshold) & (dx != 0) & (dy != 0)
    for ring in range(len(np.unique(rings))):
        eligible = np.flatnonzero(mask & (abs(rings-ring) <= settings.radial_tolerance_pixel))
        mag, angle = magnitude.flat[eligible], phase.flat[eligible]
        for left in range(len(eligible)-1):
            phase_distance = np.abs(angle[left+1:] - angle[left])
            possible = np.flatnonzero((abs(mag[left+1:]-mag[left]) <= threshold) &
                                      (abs(phase_distance-np.pi) <= settings.phase_tolerance_rad))
            if len(possible):
                keep.flat[eligible[[left, left+1+possible[0]]]] = False
    return gradient_moment(np.column_stack((dx[keep], dy[keep])), int(mask.sum()))


def extract_prepared(path, settings=Settings()):
    required = ("clean_g", "raw_h", "enhanced_a", "sky_subtracted_s", "filter_input_s",
                "mask_g", "mask_h", "mask_a", "mask_s")
    with np.load(path, allow_pickle=False) as blob:
        if str(blob["survey"].item()) != "SDSS-DR7" or str(blob["band"].item()) != "r":
            raise ValueError("Expected prepared scientific SDSS-DR7 r-band intermediates")
        arrays = {name: np.asarray(blob[name], dtype=float) for name in required}
        radial = np.asarray(blob["cumulative_flux"], dtype=float)
        scalar = {name: float(blob[name].item()) for name in
                  ("rp_pixels", "fwhm_pixels", "max_rad_pixels", "cx_pixels", "cy_pixels")}
    shape = arrays["clean_g"].shape
    if (len(shape) != 2 or min(shape) < 3 or
            any(a.shape != shape or not np.isfinite(a).all() for a in arrays.values()) or
            radial.ndim != 1 or not np.isfinite(radial).all() or
            not np.isfinite(list(scalar.values())).all()):
        raise ValueError("Invalid image intermediates, radial curve, or metadata")
    rp, fwhm, max_rad, cx, cy = scalar.values()
    if min(rp, fwhm, max_rad) <= 0 or not (0 <= cx < shape[1] and 0 <= cy < shape[0]):
        raise ValueError("Invalid pixel units or image center")
    for name in required[-4:]:
        if not np.isin(arrays[name], [0, 1]).all():
            raise ValueError(f"{name} must be binary, with 1 denoting valid pixels")
        arrays[name] = arrays[name].astype(bool)
    values = (
        concentration(radial, rp, settings.inner_fraction, settings.outer_fraction),
        asymmetry(arrays["enhanced_a"], arrays["mask_a"], fwhm),
        smoothness(arrays["sky_subtracted_s"], arrays["filter_input_s"], arrays["mask_s"],
                   max_rad, fwhm, settings),
        g2(arrays["clean_g"], arrays["mask_g"], cx, cy, settings),
        entropy(arrays["raw_h"], arrays["mask_h"], settings.entropy_bins),
    )
    flags = [f"{name}_unresolved" for name, value in zip(FEATURES, values) if not np.isfinite(value)]
    if np.isfinite(values[3]) and not 0 < values[3] < 2:
        flags.append("g2_outside_author_pipeline_range")
    return dict(zip(FEATURES, values)), flags

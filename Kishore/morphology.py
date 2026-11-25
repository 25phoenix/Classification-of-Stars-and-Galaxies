"""Deterministic morphology measurements on rendered cutouts."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi

FEATURES = (
    "concentration", "asymmetry", "smoothness", "entropy", "gini", "m20",
    "ellipticity",
)


@dataclass(frozen=True)
class Settings:
    size: int = 128
    border_fraction: float = 0.125
    detection_sigma: float = 2.5
    minimum_noise: float = 1 / 255
    central_fraction: float = 0.18
    aperture_multiplier: float = 2.5
    entropy_bins: int = 32


def intensity(image, size):
    image = np.asarray(image)
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 16:
        raise ValueError("Expected spatial dimensions >=16 and grayscale or RGB")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Expected exactly three RGB channels")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError("Expected finite real pixels")
    limit = 1 if image.dtype.kind == "f" else 255
    if image.min() < 0 or image.max() > limit:
        raise ValueError("Float pixels must be in [0,1], integers in [0,255]")
    result = image.astype(np.float64) / limit
    if result.ndim == 3:
        result = result @ np.array([0.2126, 0.7152, 0.0722])
    height, width = result.shape
    scale = size / max(height, width)
    if scale < 1:
        result = ndi.gaussian_filter(result, (1 / scale - 1) / 2)
    return ndi.zoom(result, (max(1, round(height * scale)) / height,
                             max(1, round(width * scale)) / width), order=1)


def gini(flux):
    values = np.sort(flux)
    n = len(values)
    total = values.sum()
    if n < 2 or total <= 0:
        return np.nan
    return float(np.dot(2 * np.arange(1, n + 1) - n - 1, values) / ((n - 1) * total))


def m20(flux, radius_squared):
    total = np.dot(flux, radius_squared)
    if total <= 0:
        return np.nan
    order = np.argsort(-flux, kind="stable")
    selected = order[:np.searchsorted(np.cumsum(flux[order]), 0.2 * flux.sum()) + 1]
    numerator = np.dot(flux[selected], radius_squared[selected])
    return float(np.log10(numerator / total)) if numerator > 0 else np.nan


def extract(image, settings=Settings(), diagnostic=False):
    """Return seven floats, quality flags, and optional diagnostic arrays."""
    empty = dict.fromkeys(FEATURES, np.nan)
    try:
        gray = intensity(image, settings.size)
    except (ValueError, TypeError):
        return empty, ["malformed"], None
    details = {"gray": gray} if diagnostic else None
    if np.ptp(gray) <= np.finfo(float).eps:
        return empty, ["blank"], details

    height, width = gray.shape
    y, x = np.indices(gray.shape)
    edge = max(1, round(min(gray.shape) * settings.border_fraction))
    border = (x < edge) | (y < edge) | (x >= width - edge) | (y >= height - edge)
    sky = np.median(gray[border])
    noise = max(1.4826 * np.median(np.abs(gray[border] - sky)), settings.minimum_noise)
    signal = gray - sky
    if diagnostic:
        details["signal"] = signal
    detection = ndi.gaussian_filter(signal, 1)
    components, count = ndi.label(detection > settings.detection_sigma * noise,
                                  structure=np.ones((3, 3)))
    geometric_radius = np.hypot(y - (height - 1) / 2, x - (width - 1) / 2)
    central = geometric_radius <= settings.central_fraction * min(gray.shape)
    possible = np.unique(components[central & (components > 0)])
    if not count or not len(possible):
        return empty, ["failed_segmentation", "low_signal"], details
    nearest = ndi.minimum(geometric_radius, components, possible)
    target = components == possible[np.argmin(nearest)]
    if diagnostic:
        details["target"] = target
    positive = np.maximum(signal, 0)
    weights = positive[target]
    if target.sum() < 5 or weights.sum() <= 0:
        return empty, ["unresolved"], details
    cy = np.average(y[target], weights=weights)
    cx = np.average(x[target], weights=weights)
    radius = np.hypot(y - cy, x - cx)
    aperture_radius = max(4., settings.aperture_multiplier * np.sqrt(target.sum() / np.pi))
    aperture = radius <= aperture_radius
    flags = []
    if weights.mean() < 3 * noise:
        flags.append("low_signal")
    if aperture_radius > min(cx, cy, width - 1 - cx, height - 1 - cy):
        flags.append("truncated")
    neighbors = (components > 0) & ~target
    population = aperture & ~neighbors
    light = np.where(population, positive, 0)
    flux = light[population]
    total = flux.sum()
    order = np.argsort(radius[population], kind="stable")
    cumulative = np.cumsum(flux[order])
    cumulative_unique, first = np.unique(cumulative, return_index=True)
    r20, r50, r80 = np.interp(total * np.array([.2, .5, .8]), cumulative_unique,
                              radius[population][order][first])
    concentration = 5 * np.log10(r80 / r20) if 0 < r20 <= r80 else np.nan

    # Use the flux centroid and bilinear rotation on the masked aperture.
    rotated = ndi.affine_transform(light, -np.eye(2), offset=(2 * cy, 2 * cx), order=1)
    asymmetry = np.abs(light - rotated).sum() / total
    sigma = max(1., .25 * r50)
    blurred = ndi.gaussian_filter(light, sigma)
    smoothness = np.maximum(light - blurred, 0)[population & (radius > 2 * sigma)].sum() / total
    hist, _ = np.histogram(flux / flux.max(), bins=settings.entropy_bins, range=(0, 1))
    probabilities = hist[hist > 0] / hist.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))

    yy, xx = y[population], x[population]
    dy, dx = yy - np.average(yy, weights=flux), xx - np.average(xx, weights=flux)
    cross = np.dot(flux, dy * dx) / total
    moments = np.linalg.eigvalsh([[np.dot(flux, dy ** 2) / total, cross],
                                 [cross, np.dot(flux, dx ** 2) / total]])
    ellipticity = 1 - np.sqrt(max(0, moments[0]) / moments[1]) if moments[1] > 0 else np.nan
    values = dict(zip(FEATURES, (concentration, asymmetry, smoothness, entropy,
                                gini(flux), m20(flux, dy ** 2 + dx ** 2), ellipticity)))
    if not np.isfinite(list(values.values())).all():
        flags.append("unresolved")
    if diagnostic:
        details.update(center=(cx, cy), aperture=aperture_radius, radii=(r20, r80),
                       population=population)
    return values, flags, details

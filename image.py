from PIL import Image, ImageCms
import colorsys
import numpy as np
import torch

lightroom_presets = [
    {"name": "red", "hue_min": 0, "hue_max": 20},
    {"name": "orange", "hue_min": 20, "hue_max": 45},
    {"name": "yellow", "hue_min": 45, "hue_max": 75},
    {"name": "green", "hue_min": 75, "hue_max": 150},
    {"name": "aqua", "hue_min": 150, "hue_max": 195},
    {"name": "blue", "hue_min": 195, "hue_max": 255},
    {"name": "purple", "hue_min": 255, "hue_max": 285},
    {"name": "magenta", "hue_min": 285, "hue_max": 360}
]

rgb_profile = ImageCms.getOpenProfile("./sRGB-elle-V2-srgbtrc.icc")
cmyk_profile = ImageCms.getOpenProfile("./SWOP2006_Coated3v2.icc")

cmyk_transform = ImageCms.buildTransformFromOpenProfiles(
    inputProfile=rgb_profile,
    outputProfile=cmyk_profile,
    inMode="RGB",
    outMode="CMYK",
    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC
)

srgb_transform = ImageCms.buildTransformFromOpenProfiles(
    inputProfile=cmyk_profile,
    outputProfile=rgb_profile,
    inMode="CMYK",
    outMode="RGB",
    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION | ImageCms.Flags.SOFTPROOFING
)

def rgb_to_cmyk(img: Image.Image) -> Image.Image:
    result = ImageCms.applyTransform(
        im=img,
        transform=cmyk_transform
    )

    if result is None:
        raise ValueError("Unable to convert image to cmyk")

    return result

def soft_proof(img: Image.Image) -> Image.Image:
    result = ImageCms.applyTransform(
        im=img,
        transform=srgb_transform
    )

    if result is None:
        raise ValueError("Unable to perform soft proofing")

    return result

def rgb_to_hls(r, g, b):
    return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

def hls_to_rgb(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)

def adjust_hsl_for_hue_range(h, l, s, hue_min, hue_max, h_shift=0.0, s_mult=1.0, l_offset=0.0):
    deg = h * 360
    if hue_min <= deg < hue_max:
        h = (h + h_shift) % 1.0
        s = min(max(s * s_mult, 0.0), 1.0)
        l = min(max(l + l_offset, 0.0), 1.0)
    return h, l, s

def _rgb_to_hls_torch(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # rgb: (H, W, 3) in [0, 1]
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc, _ = torch.max(rgb, dim=-1)
    minc, _ = torch.min(rgb, dim=-1)
    l = (maxc + minc) * 0.5

    delta = maxc - minc
    s = torch.zeros_like(l)
    nonzero = delta > 0

    l_le_half = l <= 0.5
    s = torch.where(
        nonzero & l_le_half,
        delta / (maxc + minc + 1e-12),
        s,
    )
    s = torch.where(
        nonzero & ~l_le_half,
        delta / (2.0 - maxc - minc + 1e-12),
        s,
    )

    h = torch.zeros_like(l)
    rc = (maxc - r) / (delta + 1e-12)
    gc = (maxc - g) / (delta + 1e-12)
    bc = (maxc - b) / (delta + 1e-12)

    h = torch.where((r == maxc) & nonzero, (bc - gc), h)
    h = torch.where((g == maxc) & nonzero, 2.0 + (rc - bc), h)
    h = torch.where((b == maxc) & nonzero, 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0

    return h, l, s


def _hls_to_rgb_torch(h: torch.Tensor, l: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # h, l, s in [0, 1]
    def _hue2rgb(m1: torch.Tensor, m2: torch.Tensor, h_: torch.Tensor) -> torch.Tensor:
        h_ = (h_ + 1.0) % 1.0
        out = torch.empty_like(h_)
        cond1 = (6.0 * h_) < 1.0
        cond2 = (2.0 * h_) < 1.0
        cond3 = (3.0 * h_) < 2.0

        out = torch.where(cond1, m1 + (m2 - m1) * 6.0 * h_, out)
        out = torch.where(~cond1 & cond2, m2, out)
        out = torch.where(~cond1 & ~cond2 & cond3, m1 + (m2 - m1) * (2.0 / 3.0 - h_) * 6.0, out)
        out = torch.where(~cond1 & ~cond2 & ~cond3, m1, out)
        return out

    m2 = torch.where(l <= 0.5, l * (1.0 + s), l + s - l * s)
    m1 = 2.0 * l - m2

    r = _hue2rgb(m1, m2, h + 1.0 / 3.0)
    g = _hue2rgb(m1, m2, h)
    b = _hue2rgb(m1, m2, h - 1.0 / 3.0)

    rgb = torch.stack([r, g, b], dim=-1)
    return rgb


def apply_hsl_offsets_torch(
    rgb: torch.Tensor,
    offsets: torch.Tensor,
    hue_ranges=lightroom_presets,
) -> torch.Tensor:
    """
    rgb: (B, 3, H, W) in [0, 1]
    offsets: (B, 8, 3) containing (h_shift, l_offset, s_offset)
             h_shift is in turns (1.0 == 360 deg)
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must be (B, 3, H, W)")
    if offsets.ndim != 3 or offsets.shape[1] != 8 or offsets.shape[2] != 3:
        raise ValueError("offsets must be (B, 8, 3)")

    rgb_bhwc = rgb.permute(0, 2, 3, 1)
    h, l, s = _rgb_to_hls_torch(rgb_bhwc)
    h_deg = h * 360.0

    h_new = h.clone()
    l_new = l.clone()
    s_new = s.clone()

    for i, rule in enumerate(hue_ranges):
        hue_min = float(rule["hue_min"])
        hue_max = float(rule["hue_max"])

        h_shift = offsets[:, i, 0].view(-1, 1, 1)
        l_offset = offsets[:, i, 1].view(-1, 1, 1)
        s_offset = offsets[:, i, 2].view(-1, 1, 1)

        mask = (h_deg >= hue_min) & (h_deg < hue_max)
        if not torch.any(mask):
            continue

        h_new = torch.where(mask, (h_new + h_shift) % 1.0, h_new)
        l_new = torch.where(mask, torch.clamp(l_new + l_offset, 0.0, 1.0), l_new)
        s_new = torch.where(mask, torch.clamp(s_new + s_offset, 0.0, 1.0), s_new)

    out_rgb = _hls_to_rgb_torch(h_new, l_new, s_new)
    out_rgb = torch.clamp(out_rgb, 0.0, 1.0)
    return out_rgb.permute(0, 3, 1, 2)


def lightroom_hsl_sim(img, hsl_ranges):
    """
    img: PIL.Image (RGB)
    hsl_ranges: list of dicts like:
        {
            "hue_min": 0,   # start of hue range (degrees)
            "hue_max": 60,  # end of hue range
            "h_shift": 0.05,  # + hue shift
            "s_mult":  1.2,   # * saturation
            "l_offset": 0.05, # + luminance
        }
    """
    arr = np.array(img)  # shape (H, W, 3)
    rgb = torch.from_numpy(arr).to(torch.float32) / 255.0

    h, l, s = _rgb_to_hls_torch(rgb)
    h_deg = h * 360.0

    h_new = h.clone()
    l_new = l.clone()
    s_new = s.clone()
    matched = torch.zeros_like(h, dtype=torch.bool)

    for rule in hsl_ranges:
        hue_min = float(rule["hue_min"])
        hue_max = float(rule["hue_max"])
        h_shift = float(rule.get("h_shift", 0.0))
        s_mult = float(rule.get("s_mult", 1.0))
        l_offset = float(rule.get("l_offset", 0.0))

        mask = (h_deg >= hue_min) & (h_deg < hue_max) & ~matched
        if not torch.any(mask):
            continue

        h_new = torch.where(mask, (h_new + h_shift) % 1.0, h_new)
        s_new = torch.where(mask, torch.clamp(s_new * s_mult, 0.0, 1.0), s_new)
        l_new = torch.where(mask, torch.clamp(l_new + l_offset, 0.0, 1.0), l_new)
        matched = matched | mask

    out_rgb = _hls_to_rgb_torch(h_new, l_new, s_new)
    out = torch.clamp(out_rgb * 255.0, 0.0, 255.0).to(torch.uint8).cpu().numpy()

    return Image.fromarray(out)

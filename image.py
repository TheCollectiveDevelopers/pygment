from PIL import Image, ImageCms, ImageDraw
from Pylette import extract_colors
import numpy as np

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

def extract_palette(img: Image.Image, num_colors: int = 15):
    palette = extract_colors(img, palette_size=num_colors)
    palette_colours = np.array([color.rgb for color in palette])
    return palette_colours

def show_palette(pixels: np.ndarray) -> None:
    swatch_size = 100
    num_colors = pixels.shape[0]
    cols = max(1, int(np.ceil(np.sqrt(num_colors))))
    rows = int(np.ceil(num_colors / cols))

    palette_img = Image.new("RGB", (swatch_size * cols, swatch_size * rows))
    draw = ImageDraw.Draw(palette_img)

    for i, color in enumerate(pixels):
        row = i // cols
        col = i % cols
        x0 = col * swatch_size
        y0 = row * swatch_size
        draw.rectangle(
            [(x0, y0), (x0 + swatch_size, y0 + swatch_size)],
            fill=tuple(color),
        )

    palette_img.show()

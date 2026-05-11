from PIL import Image
from image import lightroom_hsl_sim

img = Image.open("./dataset/pookie.jpg")

preset_hsl = [
    # Oranges (skin tones etc.)
    {
        "hue_min": 20, "hue_max": 45,
        "h_shift": 0.02,   # slightly warmer
        "s_mult": 1.15,
        "l_offset": 0.03,
    },
    # Greens
    {
        "hue_min": 80, "hue_max": 140,
        "h_shift": 0.0,
        "s_mult": 0.80,   # desaturate
        "l_offset": -0.05,
    },
    # Blues
    {
        "hue_min": 200, "hue_max": 260,
        "h_shift": 0.03,
        "s_mult": 1.25,
        "l_offset": 0.05,
    },
]

result = lightroom_hsl_sim(img, preset_hsl)
result.show()

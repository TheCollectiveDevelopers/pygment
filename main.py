from image import extract_palette, rgb_to_cmyk, soft_proof, show_palette
from PIL import Image

img = Image.open("./dataset/pookie.jpg").copy()
soft_img = soft_proof(rgb_to_cmyk(img))

pixels = extract_palette(img, num_colors=30)
soft_pixels = extract_palette(soft_img, num_colors=30)

show_palette(pixels)
show_palette(soft_pixels)

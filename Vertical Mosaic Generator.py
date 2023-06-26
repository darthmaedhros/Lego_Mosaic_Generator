import csv
import numpy as np
from PIL import Image


def euclidean_distance(color1, color2):
    """
    Calculate the Euclidean distance between two colors.
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def hex_to_rgb(hex_value):
    """
    Convert a hexadecimal color value to RGB integers.
    """
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))


def read_color_palette(csv_file):
    """
    Read the color palette from a CSV file and return it as a NumPy array.
    """
    color_palette = []

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            rgb = hex_to_rgb(row[2])
            color_palette.append(rgb)

    return np.array(color_palette)


def get_closest_color(pixel, color_palette):
    """
    Find the closest color from the given color palette to the given pixel.
    """
    distances = [euclidean_distance(pixel, color) for color in color_palette]
    closest_color_index = np.argmin(distances)
    return color_palette[closest_color_index]


def average_color(image, x, y, width, height):
    """
    Calculate the average color of a rectangular section in an image.
    """
    r_total, g_total, b_total = 0, 0, 0
    pixel_count = width * height

    for i in range(x, x + width):
        for j in range(y, y + height):
            r, g, b = image.getpixel((i, j))
            r_total += r
            g_total += g
            b_total += b

    avg_r = int(r_total / pixel_count)
    avg_g = int(g_total / pixel_count)
    avg_b = int(b_total / pixel_count)

    return avg_r, avg_g, avg_b


def process_image(image_path, color_palette, alignment):
    """
    Process the image by dividing it into a grid of 1x3 pixels and filling each section with the average color.
    """
    target_height = 380     # Should be multiple of 5 & 2. Given in Lego units.

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    aspect_ratio = width / height
    target_width = int(aspect_ratio * target_height)
    target_width = target_width - (target_width % 10)  # Ensure width is divisible by five and two. Might mess with aspect ratio.
    target_size = (target_width, target_height)
    image = image.resize(target_size)

    width, height = image.size
    new_image = Image.new("RGB", (width, height))
    original_color = Image.new("RGB", (width, height))

    if alignment == 'v':
        cell_width = 2
        cell_height = 5
    else:
        cell_width = 5
        cell_height = 2

    for x in range(0, width, cell_width):
        for y in range(0, height, cell_height):
            avg_color = average_color(image, x, y, cell_width, cell_height)
            closest_color = get_closest_color(avg_color, color_palette)
            for i in range(x, x + cell_width):
                for j in range(y, y + cell_height):
                    new_image.putpixel((i, j), tuple(closest_color))
                    original_color.putpixel((j, j), avg_color)
                    
    new_image.save("output.png", "PNG")
    original_color.save("output_original_colors.png","PNG")


# Example usage
csv_file = "colors.csv"
color_palette = read_color_palette(csv_file)
image_path = "input.png"
process_image(image_path, color_palette, 'v')
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
    return tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4))


def read_color_palette(csv_file):
    """
    Read the color palette from a CSV file and return it as a NumPy array.
    """
    color_palette = []
    color_ids = []

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            rgb = hex_to_rgb(row[2])
            color_palette.append(rgb)
            color_ids.append(row[0])

    return np.array(color_palette), color_ids


def get_closest_color(pixel, color_palette):
    """
    Find the closest color from the given color palette to the given pixel.
    """
    distances = [euclidean_distance(pixel, color) for color in color_palette]

    indices = np.argpartition(distances, 2)[:2]

    # If the two closest colors are within a certain percent of each other, choose randomly from them.
    # This helps with blending at gradual color transitions.
    if abs(distances[indices[0]] - distances[indices[1]]) <= 0.20*abs(distances[indices[0]]):
        closest_color_index = np.random.choice(indices)
    else:
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


def compare_images(original_image, color_pallete, color_ids, transformed_images):
    """
    Split the image into 10x10 sections. For each, compare the original to the transformed images, and return the closest transformed image.
    """
    width, height = original_image.size

    output_image = Image.new("RGB", (width, height))

    # Setting up the LDraw file
    ldraw_file = ""

    # Define the LDraw file header
    ldraw_file += "0 FILE Mosaic.ldr\n"
    ldraw_file += "0 Mosaic from Image\n"

    for y in range(0, height, 10):  # Iterate vertically in 10-pixel increments
        for x in range(0, width, 10):  # Iterate horizontally in 10-pixel increments
            section = original_image.crop((x, y, x + 10, y + 10))  # Crop a 10x10 section

            transformed_sections = [transformed_image.crop((x, y, x + 10, y + 10)) for transformed_image in transformed_images]

            # Calculate similarity scores between the original section and transformed versions
            similarity_scores = []
            for transformed_image in transformed_sections:
                similarity_score = np.mean((np.array(section) - np.array(transformed_image)) ** 2)
                similarity_scores.append(similarity_score)

            # Find the index of the closest transformed version
            closest_index = similarity_scores.index(min(similarity_scores))

            # Return the closest transformed version
            output_image.paste(transformed_sections[closest_index], (x, y, x+10, y+10))

            # Add elements to LDraw output file
            ldraw_file += translate_to_ldraw(closest_index, transformed_sections[closest_index], color_palette, color_ids, x, y)

    output_image.save("combined_output.png", "PNG")

    # Define the LDraw file footer
    ldraw_file += "0 NOFILE\n"

    # Save the LDraw file
    with open("mosaic.ldr", "w") as file:
        file.write(ldraw_file)


def translate_to_ldraw(mode_index, image, color_palette, color_ids, x0, y0):
    ldraw_file = ""
    color_palette = [list(line) for line in color_palette]

    # Determine mode from mode_index and ['v', 'h', 't']. Find cell_size and rotation matrix from that.
    x_mod, y_mod, z_mod = 0, 0, 0

    if mode_index == 0:
        cell_width = 2
        cell_height = 5

        rotation_matrix = "0 -1 0 1 0 0 0 0 1"

        part = 3024
        x_mod = -2      # I think the rotations change the position. This is a filler parameter.

    elif mode_index == 1:
        cell_width = 5
        cell_height = 2

        rotation_matrix = "1 0 0 0 1 0 0 0 1"

        part = 3024
        y_mod = -10
    else:
        cell_width = 5
        cell_height = 5

        rotation_matrix = "1 0 0 0 0 -1 0 1 0"

        part = "3070b"
        z_mod = -10

    # Loop over each cell. Find index of color, and use that to find id. Add piece.
    for x in range(x0, x0+10, cell_width):
        for y in range(y0, y0+10, cell_height):
            # Get color from pixel
            color = list(image.getpixel((x-x0, y-y0)))

            color_id = color_ids[color_palette.index(color)]

            used_colors.add(color_id)

            # Convert coordinates to LDraw from Lego units.

            ldraw_file += f"1 {color_id} {4*x + x_mod} {4*y + y_mod} {z_mod} {rotation_matrix} {part}.dat\n"

    return ldraw_file


def process_image(image_path, color_palette, color_ids, alignment):
    """
    Process the image by dividing it into a grid of 1x3 pixels and filling each section with the average color.
    """
    target_height = 5*48  # Should be multiple of 5 & 2. Given in Lego units.

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    aspect_ratio = width / height
    target_width = int(aspect_ratio * target_height)
    target_width = target_width - (
                target_width % 10)  # Ensure width is divisible by five and two. Might mess with aspect ratio.
    target_size = (target_width, target_height)
    image = image.resize(target_size)

    if alignment == 'v' or alignment == 'c':
        cell_width = 2
        cell_height = 5
        vertical_image, vertical_original_colors = generate_image(image, color_palette, cell_width, cell_height)

        vertical_image.save("vertical_output.png", "PNG")
        vertical_original_colors.save("vertical_original_colors.png", "PNG")

    if alignment == 'h' or alignment == 'c':
        cell_width = 5
        cell_height = 2
        horizontal_image, horizontal_original_colors = generate_image(image, color_palette, cell_width, cell_height)

        horizontal_image.save("horizontal_output.png", "PNG")
        horizontal_original_colors.save("horizontal_original_colors.png", "PNG")

    if alignment == 't' or alignment == 'c':
        cell_width = 5
        cell_height = 5
        top_down_image, top_down_original_colors = generate_image(image, color_palette, cell_width, cell_height)

        top_down_image.save("top_down_output.png", "PNG")
        top_down_original_colors.save("top_down_original_colors.png", "PNG")

    if alignment == 'c':
        compare_images(image, color_palette, color_ids, [vertical_image, horizontal_image, top_down_image])


def generate_image(image, color_pallete, cell_width, cell_height):
    """
    Generate a single image and fill with the average color.
    """
    width, height = image.size

    new_image = Image.new("RGB", (width, height))
    original_color = Image.new("RGB", (width, height))

    for x in range(0, width, cell_width):
        for y in range(0, height, cell_height):
            avg_color = average_color(image, x, y, cell_width, cell_height)
            closest_color = get_closest_color(avg_color, color_palette)
            for i in range(x, x + cell_width):
                for j in range(y, y + cell_height):
                    new_image.putpixel((i, j), tuple(closest_color))
                    original_color.putpixel((i, j), avg_color)

    return new_image, original_color

used_colors = {"0"}

# Example usage
csv_file = "restricted_colors.csv"
color_palette, color_ids = read_color_palette(csv_file)
image_path = "input/cloud-city.jpeg"
process_image(image_path, color_palette, color_ids, 'c')

print(used_colors)
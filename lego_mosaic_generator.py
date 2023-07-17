import csv
import random

import numpy as np
from PIL import Image


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


def palettise_image(image, color_palette, cell_size, weight):
    # Convert image to Numpy array
    image = np.array(image)

    # Calculate distances
    distance = np.linalg.norm(image[:, :, None] - color_palette[None, None, :], axis=3)

    # Generate image using palette. Without data type, produces noise. Mismatch with Image.fromarray.
    if weight == 0:
        palettised = np.argmin(distance, axis=2).astype(np.uint8)

    else:
        sorted_indices = np.argsort(distance)

        distance = np.where(distance == 0, 1*10**-6, distance)
        weighted_distance = np.reciprocal(np.sort(distance, axis=2)) ** weight
        # weighted_distance = np.exp(np.sort(distance, axis=2) * -weight)

        # The log function generates spotty output for the weights I tried.
        # weighted_distance = np.divide(np.log10(np.sort(distance, axis=2)), np.log10(weight))

        palettised = np.zeros([image.shape[0], image.shape[1]]).astype(np.uint8)
        for y in range(0, image.shape[0], cell_size[0]):
            for x in range(0, image.shape[1], cell_size[1]):
                palettised[y:y+cell_size[0], x:x+cell_size[1]] = random.choices(sorted_indices[y, x, :], weights=weighted_distance[y, x, :])[0]


    result = color_palette[palettised].astype(np.uint8)
    palettised_image = Image.fromarray(result, 'RGB')

    return palettised_image


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
    # output_image.show()

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


def process_image(image, given_color_palette, color_ids, alignment, weight, algorithm, target_size, width_or_height):
    """
    Process the image by dividing it into a grid of 1x3 pixels and filling each section with the average color.
    """
    global color_palette
    color_palette = given_color_palette

    width, height = image.size
    aspect_ratio = width / height

    if width_or_height == "Height":
        target_height = 5*target_size  # Should be multiple of 5 & 2. Given in Lego units.

        target_width = int(aspect_ratio * target_height)
        target_width = target_width - (
                    target_width % 10)  # Ensure width is divisible by five and two. Might mess with aspect ratio.
    elif width_or_height == "Width":
        target_width = 5 * target_size  # Should be multiple of 5 & 2. Given in Lego units.

        target_height = int(target_width / aspect_ratio)
        target_height = target_height - (
                target_height % 10)  # Ensure width is divisible by five and two. Might mess with aspect ratio.
    else:
        # Default to 48 studs tall for now
        target_height = 5*48  # Should be multiple of 5 & 2. Given in Lego units.

        target_width = int(aspect_ratio * target_height)
        target_width = target_width - (
                    target_width % 10)  # Ensure width is divisible by five and two. Might mess with aspect ratio.

    target_size = (target_width, target_height)
    image = image.resize(target_size)
    image.save("resized_image.png")

    if alignment == 'v' or alignment == 'c':
        cell_width = 2
        cell_height = 5
        vertical_image = generate_image(image, color_palette, cell_width, cell_height, weight)

        vertical_image.save("vertical_output.png", "PNG")

    if alignment == 'h' or alignment == 'c':
        cell_width = 5
        cell_height = 2
        horizontal_image = generate_image(image, color_palette, cell_width, cell_height, weight)

        horizontal_image.save("horizontal_output.png", "PNG")

    if alignment == 't' or alignment == 'c':
        cell_width = 5
        cell_height = 5
        top_down_image = generate_image(image, color_palette, cell_width, cell_height, weight)

        top_down_image.save("top_down_output.png", "PNG")

    if alignment == 'c':
        compare_images(image, color_palette, color_ids, [vertical_image, horizontal_image, top_down_image])


def generate_image(image, color_palette, cell_width, cell_height, weight):
    """
    Generate a single image and fill with the average color.
    """
    width, height = image.size

    image = np.array(image)
    new_image = image.copy()

    # new_image = Image.new("RGB", (width, height))
    # original_color = Image.new("RGB", (width, height))

    for x in range(0, width, cell_width):
        for y in range(0, height, cell_height):
            avg_color = np.mean(image[y:y+cell_height, x:x+cell_width, :], axis=(0, 1))

            new_image[y:y+cell_height, x:x+cell_width] = avg_color

    new_image = palettise_image(new_image, color_palette, [cell_height, cell_width], weight)

    return new_image


used_colors = {"0"}

# Example usage
# csv_file = "restricted_colors.csv"
# color_palette, color_ids = read_color_palette(csv_file)
#
# image_path = "input/Dunwall Clock Tower.jpg"
# image = Image.open(image_path).convert("RGB")
#
# process_image(image, color_palette, color_ids, 'c', weight=10)
#
# print(used_colors)

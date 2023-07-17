import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from lego_mosaic_generator import process_image, read_color_palette


# Function to process the image based on the selected options
def generate_mosaic():
    global processed_image, color_palette, color_ids, image

    if 'image' in globals():
        selected_option = option.get()
        weight_value = weight_slider.get()
        algorithm = dropdown.get()
        size = int(size_box.get())
        width_or_height = size_dropdown.get()

        # Perform image processing based on selected options
        # Replace this with your own image processing logic
        process_image(image, color_palette, color_ids, selected_option, weight_value, algorithm, target_size=size, width_or_height=width_or_height)


        if selected_option == 'v':
            processed_image = Image.open('vertical_output.png')
        elif selected_option == 'h':
            processed_image = Image.open('horizontal_output.png')
        elif selected_option == 't':
            processed_image = Image.open('top_down_output.png')
        else:
            processed_image = Image.open('combined_output.png')

        # Display the processed image
        display_image(processed_image, output_label)


# Function to open and display the image
def open_image():
    global image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file_path:
        image = Image.open(file_path).convert("RGB")
        display_image(image, input_label)
        process_button.config(state="normal")


def display_image(image, label):
    image_frame_width = image_frame.winfo_width()
    if image_frame_width <= 20:
        image_width, image_height = calculate_resized_dimensions(image.size, 300)
    else:
        image_width, image_height = calculate_resized_dimensions(image.size, image_frame_width)

    resized_image = image.resize((image_width, image_height))
    photo = ImageTk.PhotoImage(resized_image)
    label.config(image=photo)
    label.image = photo


# Function to calculate the resized dimensions based on the original image size and the target width
def calculate_resized_dimensions(original_size, target_width):
    original_width, original_height = original_size
    aspect_ratio = original_width / original_height
    target_height = int(target_width / aspect_ratio)

    if target_height - 4 >= image_frame.winfo_height()//2:
        target_height = image_frame.winfo_height()//2
        target_width = int(target_height * aspect_ratio)
    return target_width - 4, target_height - 4


# Create the main window
window = tk.Tk()
window.title("Image Processing GUI")

# Create a frame for the controls
control_frame = tk.Frame(window)
control_frame.pack(side="left", padx=10, pady=10)

# Create the option selection area
option_frame = tk.LabelFrame(control_frame, text="Generation Style")
option_frame.pack(pady=10)

option = tk.StringVar(value="c")

v_button = tk.Radiobutton(option_frame, text="Vertically stacked", variable=option, value="v")
v_button.pack(anchor='w')

h_button = tk.Radiobutton(option_frame, text="Horizontally stacked", variable=option, value="h")
h_button.pack(anchor='w')

t_button = tk.Radiobutton(option_frame, text="Top-down", variable=option, value="t")
t_button.pack(anchor='w')

c_button = tk.Radiobutton(option_frame, text="Best combination", variable=option, value="c")
c_button.pack(anchor='w')


# Create the output size selection area
size_frame = tk.LabelFrame(control_frame, text="Output Size")
size_frame.pack(pady=10)

size_box = tk.Entry(size_frame)
size_box.insert(0, "48")
size_box.pack()

size_dropdown = ttk.Combobox(size_frame, values=["Height", "Width"])
size_dropdown.current(0)
size_dropdown.pack()

# Create the weight slider
weight_frame = tk.LabelFrame(control_frame, text="Weight")
weight_frame.pack(pady=10)

weight_slider = tk.Scale(weight_frame, from_=0, to=25, orient="horizontal", resolution=0.1)
weight_slider.set(10)
weight_slider.pack()

# Create the dropdown menu for algorithm selection
algorithm_frame = tk.LabelFrame(control_frame, text="Algorithm")
algorithm_frame.pack(pady=10)

dropdown = ttk.Combobox(algorithm_frame, values=["Algorithm 1", "Algorithm 2", "Algorithm 3"])
dropdown.current(0)
dropdown.pack()

# Create the open button
open_button = tk.Button(control_frame, text="Open Image", command=open_image)
open_button.pack(pady=10)

# Create the process button
process_button = tk.Button(control_frame, text="Generate Mosaic", command=generate_mosaic, state='disabled')
process_button.pack(pady=10)

# Create a frame for the input and output images
image_frame = tk.Frame(window, width=300)
image_frame.pack(side="right", padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create the input image display area
input_label = tk.Label(image_frame)
input_label.pack()

# Create the output image display area
output_label = tk.Label(image_frame)
output_label.pack()


csv_file = "restricted_colors.csv"
color_palette, color_ids = read_color_palette(csv_file)


# Bind a function to window resize event
def on_window_resize(event):
    if 'image' in globals():
        display_image(image, input_label)
    if 'processed_image' in globals():
        display_image(processed_image, output_label)


image_frame.bind("<Configure>", on_window_resize)

# Start the main loop
window.mainloop()
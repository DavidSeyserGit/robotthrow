import os
from rembg import remove
from PIL import Image
import io

def remove_background(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all images in the input folder
    for filename in os.listdir(input_folder):
        # Only process image files (e.g., png, jpeg, jpg)
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            with open(input_path, 'rb') as f:
                input_image = f.read()

            # Remove background
            output_image = remove(input_image)

            # Save the result to the output folder
            with open(output_path, 'wb') as f:
                f.write(output_image)

            print(f"Processed {filename}")

if __name__ == "__main__":
    input_folder = 'extracted_frames'  # Replace with the path to your input folder
    output_folder = 'output_images'  # Replace with the path where you want to save the output images
    remove_background(input_folder, output_folder)

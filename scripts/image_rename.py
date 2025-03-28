import os
import re

def rename_frames(directory, new_prefix="image_"):
    """
    Renames frame files in a directory, extracting the number.

    Args:
        directory (str): The directory containing the frame files.
        new_prefix (str): The prefix for the new filenames.
    """

    for filename in os.listdir(directory):
        if filename.startswith("frame_") and filename.endswith(".jpg"):
            match = re.search(r"frame_(\d+).jpg", filename)
            if match:
                frame_number = match.group(1)
                new_filename = f"{new_prefix}{frame_number}.jpg"
                try:
                    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                    print(f"Renamed: {filename} -> {new_filename}")
                except FileExistsError:
                    print(f"Warning: {new_filename} already exists. Skipping {filename}")

# Example usage:
directory = "output_images" # Replace with your directory
rename_frames(directory) # will rename to image_000003.jpg, image_000010.jpg and so forth.
#rename_frames(directory, "my_frame_") # will rename to my_frame_000003.jpg, my_frame_000010.jpg etc.
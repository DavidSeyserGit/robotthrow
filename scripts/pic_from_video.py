import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extracts frames from a video and saves them as images, with an optional frame interval.

    Args:
        video_path (str): The path to the input video file.
        output_folder (str): The path to the folder where frames will be saved.
        frame_interval (int): The interval between extracted frames (e.g., 10 for every 10th frame).
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()  # Read a frame

        if not ret:  # Break the loop if no more frames are available
            break

        if frame_count % frame_interval == 0:
            # Save the frame as an image
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames to {output_folder}")

# Example usage:
video_path = 'ball-2025-03-23_14.12.49.mp4'  # Replace with the path to your video file
output_folder = 'extracted_frames' # Replace with the desired output folder name.
frame_interval = 10 # Extract every 10th frame

extract_frames(video_path, output_folder, frame_interval)
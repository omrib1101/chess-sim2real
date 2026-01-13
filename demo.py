import os
import cv2
import numpy as np
import argparse
from inference.predict import predict_board 

# ==========================================
# 1. HELPER METHODS
# ==========================================
def load_image_with_unicode(path):
    """
    Reads an image from a path that may contain non-ASCII characters (like Hebrew) 
    or spaces, which standard cv2.imread cannot handle on Windows.
    """
    try:
        # Use Python's built-in open() which handles Unicode paths correctly
        with open(path, "rb") as f:
            file_buffer = f.read()
        
        # Convert the buffer to a numpy array (uint8)
        np_array = np.frombuffer(file_buffer, dtype=np.uint8)
        
        # Decode the image from the memory buffer
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

# ==========================================
# 2. MAIN DEMO LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Chess Prediction Demo")
    # Only input is required now
    parser.add_argument("--input", required=True, help="Path to an image or a directory of images")
    args = parser.parse_args()

    # Determine if input is a file or directory
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print(f"Invalid input path: {args.input}")
        return

    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        # 1. Load image and convert to RGB array
        img_bgr = load_image_with_unicode(img_path)
        if img_bgr is None:
            print(f"Skipping {img_path}: could not read image.")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Call predict_board
        # The function handles: model loading, inference, and saving to inference/results/
        print(f"\n--- Analyzing: {os.path.basename(img_path)} ---")
        board_tensor = predict_board(img_rgb)
        
        # Output the tensor to the console
        print("Predicted Board Tensor (8x8):")
        print(board_tensor)

    print("\n" + "="*40)
    print("Demo finished successfully.")
    print("Visualizations (GIFs) are saved in: inference/results/")
    print("="*40)

if __name__ == "__main__":
    main()

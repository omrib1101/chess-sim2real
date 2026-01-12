import cv2
import numpy as np
import os

def process_and_save_view(image_path, output_folder, view_type):
    """
    Loads an image, performs shift based on view_type, 
    and saves it with a suffix in the target folder.
    """
    # 1. Load image using numpy (to support Hebrew paths)
    with open(image_path, "rb") as f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    h, w = img.shape[:2]
    shift = 20
    target_color_bgr = [89, 106, 119] # RGB(119, 106, 89)
    
    # 2. Perform the shift based on the view
    result_img = None
    
    if view_type.lower() == 'west':
        # Shift left: Crop left, Pad right
        cropped = img[:, shift:]
        padding = np.full((h, shift, 3), target_color_bgr, dtype=np.uint8)
        result_img = np.hstack((cropped, padding))
        
    elif view_type.lower() == 'east':
        # Shift right: Crop right, Pad left
        cropped = img[:, :-shift]
        padding = np.full((h, shift, 3), target_color_bgr, dtype=np.uint8)
        result_img = np.hstack((padding, cropped))
        
    elif view_type.lower() == 'overhead':
        # No shift needed for overhead, just keep original
        result_img = img
    else:
        print(f"Unknown view type: {view_type}")
        return

    # 3. Construct the new filename
    # Extract filename without extension (e.g., 'frame_002552')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Add view suffix (e.g., 'frame_002552_west.jpg')
    new_filename = f"{base_name}_{view_type.lower()}.jpg"
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_path = os.path.join(output_folder, new_filename)

    # 4. Save the result
    is_success, buffer = cv2.imencode(".jpg", result_img)
    if is_success:
        with open(output_path, "wb") as f:
            f.write(buffer)
        print(f"Saved: {output_path}")

def process_entire_folder(source_folder, output_folder, view_type):
    """
    Iterates over all images in source_folder and applies the shift logic.
    """
    # Support common image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Get list of all files in the source folder
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
    
    print(f"Starting to process {len(files)} images from: {source_folder}")
    print(f"Applying view mode: {view_type}")

    for filename in files:
        full_path = os.path.join(source_folder, filename)
        process_and_save_view(full_path, output_folder, view_type)
        
    print(f"Done! All images saved to: {output_folder}")

# ==========================================
# 2. RUNNING THE CODE
# ==========================================
if __name__ == "__main__":
    # Define your paths and view here
    SRC_DIR = r"/home/noareg/my_project/data_from_roei/new_west/tagged_images"
    OUT_DIR = r"/home/noareg/my_project/data_from_roei/new_west/croped_images"
    VIEW = "west" # Change to "east" or "overhead" as needed

    if os.path.exists(SRC_DIR):
        process_entire_folder(SRC_DIR, OUT_DIR, VIEW)
    else:

        print("Source folder not found.")

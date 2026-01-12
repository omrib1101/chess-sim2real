from PIL import Image
import os

def crop_chess_image(image_path, output_path, view_angle):

    """
    Crops the image to focus only on the board based on resolution and angle.
    """
    img = Image.open(image_path)
    
    # Dictionary to store crop coordinates [left, upper, right, lower]
    # These values need to be adjusted based on your specific Blender camera frame
    crop_configs = {
            'overhead': (296, 296, 504, 504),
            'east': (586, 296, 794, 504),
            'west': (6, 296, 214, 504)
    }

    try:
        box = crop_configs[view_angle]
        cropped_img = img.crop(box)
        cropped_img.save(output_path)
    except KeyError:
        print(f"Error: No configuration")
        
        
folders = [
    (r"C:\Users\Yael\Noa\קורסים\למידה עמוקה\final project\syntetic_from_random\original_images",
     r"C:\Users\Yael\Noa\קורסים\למידה עמוקה\final project\syntetic_from_random\croped_images"),
    (r"C:\Users\Yael\Noa\קורסים\למידה עמוקה\final project\syntetic_from_excel\original_images",
     r"C:\Users\Yael\Noa\קורסים\למידה עמוקה\final project\syntetic_from_excel\croped_images")
]

for src_folder, dst_folder in folders:
    # Create destination folder if it doesn't exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"Created folder: {dst_folder}")

    # Iterate through all files in the source folder
    for filename in os.listdir(src_folder):
        if filename.endswith(".png"):
            # Extract view_angle from filename (e.g., '326_west.png' -> 'west')
            try:
                # Remove extension and split by underscore
                name_part = filename.replace(".png", "")
                view_angle = name_part.split("_")[-1].lower()
                
                full_src_path = os.path.join(src_folder, filename)
                full_dst_path = os.path.join(dst_folder, filename)
                
                # Perform the crop
                crop_chess_image(full_src_path, full_dst_path, view_angle)
                
            except Exception as e:
                print(f"Could not process {filename}: {e}")


print("\n--- All images processed successfully ---")

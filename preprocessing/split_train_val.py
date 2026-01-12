import os
import shutil
import random
from collections import defaultdict

def split_to_validation(train_dir, val_base_dir, val_ratio=0.2):
    """
    Moves 20% of images from the train directory to a validation directory,
    maintaining class distribution and organizing them into subfolders.
    """
    # 1. Collect all files in the train directory
    # (Assuming all images are currently directly inside train_dir or in its subfolders)
    all_files = []
    for root, dirs, files in os.walk(train_dir):
        for filename in files:
            if filename.endswith(".png"):
                all_files.append((filename, os.path.join(root, filename)))

    if not all_files:
        print("No .png files found in the source directory.")
        return

    # 2. Group files by label (the last letter/word before .png)
    label_to_files = defaultdict(list)
    for filename, full_path in all_files:
        try:
            # Format: num_view_index_label.png -> label
            label = filename.replace(".png", "").split("_")[-1]
            label_to_files[label].append((filename, full_path))
        except IndexError:
            continue

    print(f"Starting split... Found {len(all_files)} total images across {len(label_to_files)} classes.")

    # 3. Process each class: move 20% to validation and organize train into subfolders
    for label, files in label_to_files.items():
        random.shuffle(files)
        
        # Calculate how many to move to validation
        val_count = int(len(files) * val_ratio)
        val_files = files[:val_count]
        remaining_train_files = files[val_count:]

        # Create validation subfolder for this label
        val_label_dir = os.path.join(val_base_dir, label)
        if not os.path.exists(val_label_dir):
            os.makedirs(val_label_dir)

        # Create train subfolder for this label (to organize the source folder)
        train_label_dir = os.path.join(train_dir, label)
        if not os.path.exists(train_label_dir):
            os.makedirs(train_label_dir)

        # Move files to validation
        for filename, src_path in val_files:
            shutil.move(src_path, os.path.join(val_label_dir, filename))

        # Move remaining files into their respective label subfolder within train
        # (This cleans up the train folder so it has the same structure as validation)
        for filename, src_path in remaining_train_files:
            dest_path = os.path.join(train_label_dir, filename)
            # Avoid moving if the file is already in the correct subfolder
            if src_path != dest_path:
                shutil.move(src_path, dest_path)

        print(f"Class '{label}': Moved {len(val_files)} to validation, {len(remaining_train_files)} stayed in train.")

    print("\nDone! Your data is now organized into subfolders by piece type.")

# --- Paths Configuration ---
train_path = r"/home/noareg/my_project/data_for_train/train2/train"
val_path = r"/home/noareg/my_project/data_for_train/train2/validation"

# Execute
split_to_validation(train_path, val_path, val_ratio=0.2)

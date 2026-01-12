import pandas as pd
import subprocess
import os
import csv
import shutil

from datetime import datetime
start = datetime.now()

# Local paths
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BASE_PATH = r"C:\Users\Yael\Noa\קורסים\למידה עמוקה\final project"
INPUT_EXCEL = os.path.join(BASE_PATH, r"data\all\all_fens.csv")
OUTPUT_ROOT = os.path.join(BASE_PATH, "syntetic_from_excel")
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
CSV_DIR = os.path.join(OUTPUT_ROOT, "csv")
METADATA_FILE = os.path.join(CSV_DIR, "fens_to_images.csv")
RENDERS_SOURCE = r"C:\renders"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def reverse_fen(fen):
    return fen[::-1]

def run_blender(fen, view):
    cmd = [
        BLENDER_PATH,
        "chess-set.blend", 
        "--background",
        "--python", "chess_position_api_v2.py",
        "--",
        "--fen", fen,
        "--view", view,
        "--resolution", "800",
        "--samples", "128"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL)

# ---Start of process---
# Reading FENs from the given excel
df = pd.read_csv(INPUT_EXCEL, header=None)
metadata = []

for index, row in df.iterrows():
    original_fen = str(row[0])
    row_number = index + 1 
    
    # Mixing white and black views (for different angles of pieces)
    if row_number <= 261:
        current_view = 'white'
        fen_to_save = original_fen
    else:
        current_view = 'black'
        fen_to_save = reverse_fen(original_fen)
    
    
    run_blender(original_fen, current_view)
    
    # Dealing with the 3 pictures genereted for each FEN
    views_generated = ['overhead', 'east', 'west']
    for v in views_generated:
        # The blender saves the output in a temp folder called "renders"
        if current_view=='white':
            source_img = f"1_{v}.png" if v == 'overhead' else f"{'2' if v=='east' else '3'}_{v}.png"
        else:
            source_img = f"1_{v}.png" if v == 'overhead' else f"{'2' if v=='west' else '3'}_{v}.png"
        
        source_img=os.path.join(RENDERS_SOURCE, source_img)
        
        # Creating the new file name: {row_number}_{v}
        new_filename = f"{row_number}_{v}.png"
        target_path = os.path.join(IMAGES_DIR, new_filename)
        
        if os.path.exists(source_img):
            shutil.move(source_img, target_path)
            metadata.append({
                'image_name': f"{row_number}_{v}",
                'fen': fen_to_save
            })
        else:
            print(f"Warning: Could not find image for view {v} in row {row_number} at {RENDERS_SOURCE}")

# Saving the metadata in the excel output file
meta_df = pd.DataFrame(metadata)
meta_df.to_csv(METADATA_FILE, index=False, header=['image_name', 'fen'])

end = datetime.now()
print("Total runtime:", end - start)

print(f"finished process. {len(metadata)} images were created and saved in {IMAGES_DIR}")

import pandas as pd
import subprocess
import os
import random
import shutil

from datetime import datetime
start_time = datetime.now()

# --- Paths Configuration ---
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BASE_PATH = r"C:\Users\Yael\Noa\קורסים\למידה עמוקה\final project"
# New root folder for random generation
OUTPUT_ROOT = os.path.join(BASE_PATH, "syntetic_from_random")
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
CSV_DIR = os.path.join(OUTPUT_ROOT, "csv")
METADATA_FILE = os.path.join(CSV_DIR, "fens_to_images.csv")
RENDERS_SOURCE = r"C:\renders"

# Create folder structure
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def generate_managed_fen(density_type="dense"):
    """
    Generates a FEN based on specific inventory limits and color balance.
    No 'P' (pawns) used as requested.
    """
    # Define EXACT inventory from your Blender file
    white_inv = ['K']*6 + ['Q']*6 + ['R']*4 + ['B']*4 + ['N']*4 #24
    black_inv = ['k']*6 + ['q']*6 + ['r']*4 + ['b']*4 + ['n']*4 #24
    
    # Set piece count based on density
    if density_type == "dense":
        # Dense: 32 to 48 pieces (50% to 75% board occupancy)
        total_pieces = random.randint(32, 48)
    else:
        # Sparse: 3 to 12 pieces
        total_pieces = random.randint(3, 12)

    # Balance colors 50/50
    num_white = total_pieces // 2
    num_black = total_pieces - num_white
    
    # Sample pieces safely based on inventory size
    chosen_pieces = random.sample(white_inv, num_white) + random.sample(black_inv, num_black)
    
    random.shuffle(chosen_pieces)
    
    # Place on 8x8 board
    board = [['' for _ in range(8)] for _ in range(8)]
    squares = [(r, c) for r in range(8) for c in range(8)]
    chosen_squares = random.sample(squares, len(chosen_pieces))
    
    for i, (r, c) in enumerate(chosen_squares):
        board[r][c] = chosen_pieces[i]
        
    # Build FEN string
    fen_rows = []
    for row in board:
        empty = 0
        res = ""
        for cell in row:
            if cell == '': empty += 1
            else:
                if empty > 0:
                    res += str(empty)
                    empty = 0
                res += cell
        if empty > 0: 
            res += str(empty)
        fen_rows.append(res)
    
    # Return ONLY the position part of the FEN if that's what your API expects
    return "/".join(fen_rows)

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
    # Suppressing output to keep console clean
    subprocess.run(cmd, stdout=subprocess.DEVNULL)

# --- Start Process (225 Boards) ---
metadata = []

for i in range(1, 226):
    # 1. Logic for Density
    if i <= 150:
        density = "dense"  
    else: 
        density = "sparse"
    
    # 2. Logic for View and FEN storage (Following your requested ranges)
    # Range 1-75: White View | 76-150: Black View
    # Range 151-188: White View | 189-225: Black View
    if (1 <= i <= 75) or (151 <= i <= 188):
        current_view = 'white'
        generated_fen = generate_managed_fen(density)
        fen_for_csv = generated_fen # Keep as is
    else:
        current_view = 'black'
        generated_fen = generate_managed_fen(density)
        fen_for_csv = reverse_fen(generated_fen) # Stored reversed as in your original code
    
    # 3. Call Blender
    run_blender(generated_fen, current_view)
    
    # 4. Process the 3 generated images (overhead, east, west)
    views_generated = ['overhead', 'east', 'west']
    for v in views_generated:
        if current_view == 'white':
            source_img = f"1_{v}.png" if v == 'overhead' else f"{'2' if v=='east' else '3'}_{v}.png"
        else:
            # Black view logic for source filenames as per your provided code
            source_img = f"1_{v}.png" if v == 'overhead' else f"{'2' if v=='west' else '3'}_{v}.png"
        
        source_path = os.path.join(RENDERS_SOURCE, source_img)
        new_filename = f"{i}_{v}.png"
        target_path = os.path.join(IMAGES_DIR, new_filename)
        
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            metadata.append({
                'image_name': f"{i}_{v}",
                'fen': fen_for_csv
            })
        else:
            print(f"Warning: Could not find image for view {v} in row {i} at {RENDERS_SOURCE}")
    if i==1:
        break
# Save metadata
meta_df = pd.DataFrame(metadata)
meta_df.to_csv(METADATA_FILE, index=False, header=['image_name', 'fen'])

print(f"Total runtime: {datetime.now() - start_time}")
print(f"finished process. {len(metadata)} images were created and saved in {IMAGES_DIR}")
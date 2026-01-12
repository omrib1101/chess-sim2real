import os
import pandas as pd
from PIL import Image

def parse_fen_to_labels(fen):
    rows = fen.split('/')
    full_labels = []
    for row in rows:
        for char in row:
            if char.isdigit():
                full_labels.extend(['empty'] * int(char))
            else:
                full_labels.append(char)
    return full_labels

def split_board_to_squares(src_folder, dst_folder, csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Migrate 'to_frame' column to string for easy matching
    df['to_frame'] = df['to_frame'].astype(str)
    fen_dict = pd.Series(df.fen.values, index=df.to_frame).to_dict()

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        if not filename.endswith(".jpg"):
            continue

        # --- Data extraction from filename ---
        # Example filename: frame_000200_west.jpg
        name_part = filename.replace(".jpg", "") # frame_000200_west
        parts = name_part.split('_') # ['frame', '000200', 'west']
        
        raw_number = parts[1] # "000200"
        view = parts[2]       # "west"
        
        # Removing leading zeros from the number to match the CSV (e.g., "000200" -> "200")
        to_frame = raw_number.lstrip('0')
        if to_frame == "": to_frame = "0" # Edge case if the number is 000

        img_path = os.path.join(src_folder, filename)
        img = Image.open(img_path)
        w, h = img.size
        square_w, square_h = w // 8, h // 8

        # Retrieve FEN
        if to_frame in fen_dict:
            fen = fen_dict[to_frame]
            labels = parse_fen_to_labels(fen)
        else:
            print(f"Warning: No FEN found for frame number '{to_frame}' (file: {filename}). Skipping...")
            continue

        # Split image into 64 squares
        count = 1
        for row_idx in range(8):
            for col_idx in range(8):
                label = labels[count-1]
                
                left = col_idx * square_w
                upper = row_idx * square_h
                right = left + square_w
                lower = upper + square_h
                
                square_img = img.crop((left, upper, right, lower))
                
                # --- Creating new filename  ---
                # {to_frame}_{view}_{count}_{label}.png
                new_filename = f"{to_frame}_{view}_{count}_{label}.png"
                
                square_img.save(os.path.join(dst_folder, new_filename))
                count += 1
                
        print(f"Successfully processed: {filename} as frame {to_frame}")

# Run (each time with updated paths)
src = r"/home/noareg/my_project/data_from_roei/new_overhead/croped_images"
dst = r"/home/noareg/my_project/data_from_roei/new_overhead/croped_squares"
csv = r"/home/noareg/my_project/data_from_roei/new_images.csv"

split_board_to_squares(src, dst, csv)

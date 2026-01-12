import os
import cv2
import torch
import numpy as np
import argparse
import requests
from urllib.parse import quote
from inference.predict import predict_board 

# ==========================================
# 1. LICHESS UTILS 
# ==========================================
def save_board_from_lichess(fen: str, out_path: str, color: str = "white"):
    """
    Downloads a chessboard image from Lichess based on FEN.
    """
    # Lichess expects spaces replaced with underscores
    fen_for_url = fen.replace(" ", "_")
    url = (
        "https://lichess1.org/export/fen.gif"
        f"?fen={quote(fen_for_url, safe='/_')}"
        f"&color={color}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading from Lichess: {e}")
        return False

# ==========================================
# 2. CONVERSION: TENSOR (0-12) TO FEN
# ==========================================
def tensor_to_fen(board_tensor: torch.Tensor) -> str:
    """
    Converts the 8x8 Tensor (values 0-12) back to a FEN string.
    Mapping: 0-5 White (P,N,B,R,Q,K), 6-11 Black (p,n,b,r,q,k), 12 Empty.
    """
    # Inverse mapping from your piece values
    mapping = {
        0:'P', 1:'N', 2:'B', 3:'R', 4:'Q', 5:'K',
        6:'p', 7:'n', 8:'b', 9:'r', 10:'q', 11:'k'
    }
    
    fen_rows = []
    board = board_tensor.cpu().numpy()
    
    for r in range(8):
        empty_count = 0
        row_str = ""
        for c in range(8):
            val = board[r, c]
            if val == 12:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += mapping[val]
        if empty_count > 0:
            row_str += str(empty_count)
        fen_rows.append(row_str)
    
    return "/".join(fen_rows)

# ==========================================
# 3. MAIN DEMO LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Chess Prediction Demo")
    parser.add_argument("--input", required=True, help="Path to an image or a directory of images")
    parser.add_argument("--output_dir", required=True, help="Directory to save the results")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Determine if input is a file or directory
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print("Invalid input path.")
        return

    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        # 1. Load image and convert to RGB array
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Skipping {img_path}: could not read image.")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Call the mandatory predict_board function (receives array, returns Tensor)
        board_tensor = predict_board(img_rgb)

        # 3. Convert Tensor to FEN for Lichess
        predicted_fen = tensor_to_fen(board_tensor)

        # 4. Save visualization to output directory
        out_filename = f"pred_{os.path.splitext(os.path.basename(img_path))[0]}.gif"
        out_path = os.path.join(args.output_dir, out_filename)
        
        success = save_board_from_lichess(predicted_fen, out_path)
        if not success:
            print(f"Failed to save result to: {out_path}")

    print("\nDemo finished successfully.")

if __name__ == "__main__":
    main()
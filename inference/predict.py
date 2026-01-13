import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import urllib.request
import requests
from urllib.parse import quote
import time

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class ChessMultiTaskModel(nn.Module):
    def __init__(self):
        super(ChessMultiTaskModel, self).__init__()
        resnet = models.resnet18(weights=None) 
        num_ftrs = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.empty_head = nn.Linear(num_ftrs, 2)
        self.color_head = nn.Linear(num_ftrs, 2)
        self.piece_head = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.empty_head(x), self.color_head(x), self.piece_head(x)

# ==========================================
# 2. CONFIGURATION & GLOBAL INITIALIZATION
# ==========================================
# Relative path to the model
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.normpath(os.path.join(current_dir, "..", "checkpoints", "combined.pth"))

# Mapping:
# White: P=0, N=1, B=2, R=3, Q=4, K=5 | Black: p=6, n=7, b=8, r=9, q=10, k=11 | Empty: 12
# Our Model internal piece order: B, K, N, P, Q, R (indices 0-5)
INTERNAL_TO_PDF_PIECE = {0: 2, 1: 5, 2: 1, 3: 0, 4: 4, 5: 3}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model globally to avoid repeated overhead in demo.py
_GLOBAL_MODEL = None

def ensure_model_exists():
    """
    Checks if the model weights file exists locally. 
    If not, downloads it from the specified GitHub Release URL.
    """
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Weights not found at {MODEL_WEIGHTS_PATH}. Downloading from GitHub Release...")
        
        # IMPORTANT: Replace this URL with your actual GitHub Release direct link
        url = "https://github.com/omrib1101/chess-sim2real/releases/download/v1.0/combined.pth"
        
        # Ensure the target directory (checkpoints/) exists
        os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
        
        try:
            # Download the file from the URL and save it to MODEL_WEIGHTS_PATH
            urllib.request.urlretrieve(url, MODEL_WEIGHTS_PATH)
            print("Download complete!")
        except Exception as e:
            # Raise an informative error if the download fails (e.g., no internet connection)
            raise Exception(f"Failed to download model weights. Please check your internet connection. Error: {e}")

def _get_model():
    """
    Singleton-pattern function to load and return the model.
    Ensures weights are downloaded and loaded into memory only once.
    """
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        # Step 1: Verify the weights file is present locally
        ensure_model_exists()
        
        # Step 2: Initialize the model architecture and move to the target device (CPU/GPU)
        _GLOBAL_MODEL = ChessMultiTaskModel().to(DEVICE)
        
        # Step 3: Load the state dictionary. 
        # Note: weights_only=False is used to support models saved with full class structures 
        # or in older PyTorch versions.
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        _GLOBAL_MODEL.load_state_dict(state_dict)
        
        # Step 4: Set the model to evaluation mode
        _GLOBAL_MODEL.eval()
        
    return _GLOBAL_MODEL

# ==========================================
# 3. HELPER METHODS (from your image_to_fen)
# ==========================================
def get_three_views_from_array(img_array):
    """
    Creates three views (Overhead, West, East) from a numpy RGB array.
    Uses padding with a specific table color to maintain image size.
    """
    # 1. Convert RGB (input) to BGR (for OpenCV processing)
    # OpenCV uses BGR, but your input from the system is RGB.
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    h, w = img_bgr.shape[:2]
    shift = 20
    target_color_bgr = [89, 106, 119] # Fixed padding color

    # 2. Overhead View (The original image)
    # We convert the original array directly to PIL
    views = [Image.fromarray(img_array)]

    # 3. West View (Shifted left, padded on the right)
    # img_bgr[:, shift:] "moves" the image left by removing the left 20 pixels
    west_img = img_bgr[:, shift:]
    # Create the padding block (20 pixels wide, same height)
    pad_w = np.full((h, shift, 3), target_color_bgr, dtype=np.uint8)
    # Stack the shifted image and the pad horizontally
    # Then convert BGR -> RGB -> PIL
    west_view_combined = np.hstack((west_img, pad_w))
    west_view_rgb = cv2.cvtColor(west_view_combined, cv2.COLOR_BGR2RGB)
    views.append(Image.fromarray(west_view_rgb))

    # 4. East View (Shifted right, padded on the left)
    # img_bgr[:, :-shift] "moves" the image right by removing the right 20 pixels
    east_img = img_bgr[:, :-shift]
    # Create the padding block
    pad_e = np.full((h, shift, 3), target_color_bgr, dtype=np.uint8)
    # Stack the pad first, then the image
    # Then convert BGR -> RGB -> PIL
    east_view_combined = np.hstack((pad_e, east_img))
    east_view_rgb = cv2.cvtColor(east_view_combined, cv2.COLOR_BGR2RGB)
    views.append(Image.fromarray(east_view_rgb))

    return views

def _internal_tensor_to_fen(board_tensor: torch.Tensor) -> str:
    """ Converts 8x8 Tensor to FEN string for Lichess """
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

def _internal_save_to_lichess(fen: str, out_path: str):
    """ Downloads GIF from Lichess """
    fen_for_url = fen.replace(" ", "_")
    url = f"https://lichess1.org/export/fen.gif?fen={quote(fen_for_url, safe='/_')}&color=white"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Note: Could not save Lichess visualization: {e}")

# ==========================================
# 4. REQUIRED API FUNCTION
# ==========================================
def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Input: image (np.ndarray) - RGB Chessboard image
    Output: torch.Tensor - 8x8 matrix (int64) 
    """
    model = _get_model()
    
    # 1. Create the views to help with piece detection
    views = get_three_views_from_array(image)
    
    # 2. Define Transform
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    board_matrix = np.zeros((8, 8), dtype=np.int64)

    with torch.no_grad():
        for r in range(8):
            for c in range(8):
                # We sum probabilities across views to reach a consensus
                sum_empty = torch.zeros(2).to(DEVICE)
                sum_color = torch.zeros(2).to(DEVICE)
                sum_piece = torch.zeros(6).to(DEVICE)

                for view in views:
                    w_v, h_v = view.size
                    sw, sh = w_v // 8, h_v // 8
                    # Crop square (c, r)
                    square_crop = view.crop((c * sw, r * sh, (c + 1) * sw, (r + 1) * sh))
                    input_tensor = preprocess(square_crop).unsqueeze(0).to(DEVICE)
                    
                    o_empty, o_color, o_piece = model(input_tensor)
                    sum_empty += torch.softmax(o_empty, dim=1)[0]
                    sum_color += torch.softmax(o_color, dim=1)[0]
                    sum_piece += torch.softmax(o_piece, dim=1)[0]

                # Decision Step
                # If index 1 of empty_head is higher -> square is empty
                if torch.argmax(sum_empty).item() == 1:
                    board_matrix[r, c] = 12
                else:
                    is_white = torch.argmax(sum_color).item() == 1
                    p_idx_internal = torch.argmax(sum_piece).item()
                    
                    # Convert to PDF piece index
                    piece_val = INTERNAL_TO_PDF_PIECE[p_idx_internal]
                    
                    if not is_white:
                        piece_val += 6 # Black offset
                    
                    board_matrix[r, c] = piece_val

    # Ensure output is a torch.Tensor as requested

    result_tensor = torch.from_numpy(board_matrix).to(torch.int64).cpu()
    try:
        # Determine the results directory: inference/results/
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_script_dir, "results")
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Create FEN
        current_fen = _internal_tensor_to_fen(result_tensor)
        
        # Generate a unique filename using timestamp with milliseconds
        # %f gives microseconds, we take the first 3 digits for milliseconds
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        milli = int(time.time() * 1000) % 1000
        output_path = os.path.join(results_dir, f"prediction_{timestamp}_{milli:03d}.gif")
        
        _internal_save_to_lichess(current_fen, output_path)
        print(f"Visualization saved to: {output_path}")
    except Exception as e:
        print(f"Auto-save failed: {e}")

    return result_tensor



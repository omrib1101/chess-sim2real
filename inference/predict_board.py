import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

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
# PLACEHOLDER: Update this to your weight file path in your Git structure
MODEL_WEIGHTS_PATH = "models/chess_multitask_combine.pth"

# Mapping:
# White: P=0, N=1, B=2, R=3, Q=4, K=5 | Black: p=6, n=7, b=8, r=9, q=10, k=11 | Empty: 12
# Our Model internal piece order: B, K, N, P, Q, R (indices 0-5)
INTERNAL_TO_PDF_PIECE = {0: 2, 1: 5, 2: 1, 3: 0, 4: 4, 5: 3}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model globally to avoid repeated overhead in demo.py
_GLOBAL_MODEL = None

def _get_model():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = ChessMultiTaskModel().to(DEVICE)
        _GLOBAL_MODEL.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
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
    target_color_bgr = [89, 106, 119] # The table color used in your project

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
    return torch.from_numpy(board_matrix)
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

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
# 2. UTILS
# ==========================================
def get_three_views(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    shift = 20
    target_color_bgr = [89, 106, 119] # RGB(119, 106, 89)

    views = [pil_img] # Overhead

    # West View
    west_img = img_bgr[:, shift:]
    pad_w = np.full((h, shift, 3), target_color_bgr, dtype=np.uint8)
    west_view = cv2.cvtColor(np.hstack((west_img, pad_w)), cv2.COLOR_BGR2RGB)
    views.append(Image.fromarray(west_view))

    # East View
    east_img = img_bgr[:, :-shift]
    pad_e = np.full((h, shift, 3), target_color_bgr, dtype=np.uint8)
    east_view = cv2.cvtColor(np.hstack((pad_e, east_img)), cv2.COLOR_BGR2RGB)
    views.append(Image.fromarray(east_view))

    return views

def labels_to_fen(board_labels):
    fen = ""
    for r in range(8):
        empty_count = 0
        for c in range(8):
            label = board_labels[r * 8 + c]
            if label == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += label
        if empty_count > 0:
            fen += str(empty_count)
        if r < 7: fen += "/"
    return fen

# ==========================================
# 3. CORE FUNCTION FOR EXPORT
# ==========================================
def get_fen_from_image(image_path, model_path):
    """
    Given an image path and a trained 3-head model path,
    returns the predicted FEN string using multi-view voting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load model
    model = ChessMultiTaskModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    PIECE_MAP = {0: 'B', 1: 'K', 2: 'N', 3: 'P', 4: 'Q', 5: 'R'}
    original_img = Image.open(image_path).convert("RGB")
    views = get_three_views(original_img)
    board_labels = []
    
    with torch.no_grad():
        for i in range(64):
            r, c = divmod(i, 8)
            sum_empty = torch.zeros(2).to(device)
            sum_color = torch.zeros(2).to(device)
            sum_piece = torch.zeros(6).to(device)

            for view in views:
                w_img, h_img = view.size
                sw, sh = w_img // 8, h_img // 8
                crop = view.crop((c * sw, r * sh, (c + 1) * sw, (r + 1) * sh))
                input_tensor = preprocess(crop).unsqueeze(0).to(device)
                
                o_empty, o_color, o_piece = model(input_tensor)
                sum_empty += torch.softmax(o_empty, dim=1)[0]
                sum_color += torch.softmax(o_color, dim=1)[0]
                sum_piece += torch.softmax(o_piece, dim=1)[0]

            if torch.argmax(sum_empty).item() == 1:
                board_labels.append("empty")
            else:
                c_idx = torch.argmax(sum_color).item()
                p_idx = torch.argmax(sum_piece).item()
                char = PIECE_MAP[p_idx]
                final_char = char if c_idx == 1 else char.lower()
                board_labels.append(final_char)

    return labels_to_fen(board_labels)

# ==========================================
# 4. MAIN FOR TESTING
# ==========================================
if __name__ == "__main__":
    # Example usage:
    my_image = "/home/noareg/my_project/single_test/frame_000036.jpg"
    my_model = "/home/noareg/my_project/codes/model/zero_shot_model.pth"

    if os.path.exists(my_image) and os.path.exists(my_model):
        result_fen = get_fen_from_image(my_image, my_model)
        print(f"\nPredicted FEN:\n{result_fen}")
    else:
        print("Error: Please check your file paths.")
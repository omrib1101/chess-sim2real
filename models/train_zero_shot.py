import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time
import copy
from tqdm import tqdm

# ==========================================
# 1. DATASET CLASS
# ==========================================
class ChessMultiTaskDataset(Dataset):
    """
    Custom Dataset to map folder names to 3 labels: Is_Empty, Color, and Piece_Type.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Mapping: (is_empty [0:Full, 1:Empty], color [0:B, 1:W, 2:N/A], piece [0:B, 1:K, 2:N, 3:P, 4:Q, 5:R, 6:N/A])
        self.class_map = {
            'black_B': (0, 0, 0), 'black_K': (0, 0, 1), 'black_N': (0, 0, 2),
            'black_P': (0, 0, 3), 'black_Q': (0, 0, 4), 'black_R': (0, 0, 5),
            'white_B': (0, 1, 0), 'white_K': (0, 1, 1), 'white_N': (0, 1, 2),
            'white_P': (0, 1, 3), 'white_Q': (0, 1, 4), 'white_R': (0, 1, 5),
            'empty':   (1, 2, 6) 
        }

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path): continue
            labels = self.class_map.get(class_name)
            if labels is None: continue
            
            for img_name in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, img_name), labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (l_empty, l_color, l_piece) = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(l_empty), torch.tensor(l_color), torch.tensor(l_piece)

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class ChessMultiTaskModel(nn.Module):
    def __init__(self):
        super(ChessMultiTaskModel, self).__init__()
        resnet = models.resnet18(weights='DEFAULT')
        num_ftrs = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Task Heads
        self.empty_head = nn.Linear(num_ftrs, 2)
        self.color_head = nn.Linear(num_ftrs, 2)
        self.piece_head = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.empty_head(features), self.color_head(features), self.piece_head(features)

# ==========================================
# 3. CONDITIONAL LOSS FUNCTION
# ==========================================
def multitask_loss_fn(outputs, t_empty, t_color, t_piece, criterion):
    o_empty, o_color, o_piece = outputs
    loss_empty = criterion(o_empty, t_empty)
    
    # Conditional masking for pieces and color
    mask = (t_empty == 0)
    loss_color, loss_piece = 0, 0
    if mask.any():
        loss_color = criterion(o_color[mask], t_color[mask])
        loss_piece = criterion(o_piece[mask], t_piece[mask])
        
    # Weighted sum of losses
    return 0.5*loss_empty + loss_color + 2*loss_piece

# ==========================================
# 4. TRAINING FUNCTION
# ==========================================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=15, patience=3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}\n' + '-'*10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            pbar = tqdm(dataloaders[phase], desc=f"Phase {phase}")

            for inputs, t_empty, t_color, t_piece in pbar:
                inputs = inputs.to(device)
                t_empty, t_color, t_piece = t_empty.to(device), t_color.to(device), t_piece.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = multitask_loss_fn(outputs, t_empty, t_color, t_piece, criterion)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'validation':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/noareg/my_project/data_for_train/train1"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
            transforms.RandomRotation(10), # Pieces aren't always perfectly centered
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: ChessMultiTaskDataset(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'validation']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

    model = ChessMultiTaskModel().to(device)
    
    # Differential Learning Rates (unfreeze layer4 and fc)
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.backbone[7].parameters():  # layer4
        param.requires_grad = True

    for head in [model.empty_head, model.color_head, model.piece_head]:
        for param in head.parameters():
            param.requires_grad = True


    optimizer = optim.Adam([
        {'params': model.backbone[7].parameters(), 'lr': 3e-5}, # Backbone[7] is layer4
        {'params': model.empty_head.parameters(), 'lr': 3e-4},
        {'params': model.color_head.parameters(), 'lr': 3e-4},
        {'params': model.piece_head.parameters(), 'lr': 5e-4}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    criterion = nn.CrossEntropyLoss()

    trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device)
    torch.save(trained_model.state_dict(), 'zero_shot_model.pth')

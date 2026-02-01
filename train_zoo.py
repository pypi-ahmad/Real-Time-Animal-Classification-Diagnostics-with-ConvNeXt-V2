import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

# --- Configuration ---
DATA_DIR = 'raw-img'  # Directory containing the dataset
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 10
PATIENCE = 3
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable Cudnn Benchmark for consistent input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Label Mapping (Italian -> English)
LABEL_MAPPING = {
    'cane': 'Dog', 
    'cavallo': 'Horse', 
    'elefante': 'Elephant', 
    'farfalla': 'Butterfly', 
    'gallina': 'Chicken', 
    'gatto': 'Cat', 
    'mucca': 'Cow', 
    'pecora': 'Sheep', 
    'ragno': 'Spider', 
    'scoiattolo': 'Squirrel'
}

# --- Data Helpers ---

class TransformedSubset(Dataset):
    """
    Wrapper to apply specific transforms to a Subset of a Dataset.
    
    Args:
        subset (torch.utils.data.Subset): The subset of the original dataset.
        transform (callable, optional): A function/transform to apply to the image.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# --- Main Training Script ---

def main():
    """
    Main execution entry point for training the ZooStation AI model.
    
    Pipeline:
    1. Sets up data transforms (Data Augmentation for training, Normalization for eval).
    2. Loads the dataset from 'raw-img' and splits it into Train (70%), Val (15%), Test (15%).
    3. Initializes the ConvNeXt V2 Tiny model (Pretrained on ImageNet).
    4. Trains the model using Mixed Precision (AMP) for efficiency.
    5. Evaluates on Validation set with Early Stopping.
    6. Saves the best model as 'zoo_bundle.pth'.
    """
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found. Please ensure 'raw-img' exists.")
        return

    # 1. Data Pipeline
    print("Setting up data pipeline...")
    
    # Transforms
    # Standard ImageNet mean/std used for Transfer Learning
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load Base Dataset (No transforms yet)
    base_dataset = datasets.ImageFolder(root=DATA_DIR)
    
    # Map classes to English for artifacts
    original_classes = base_dataset.classes
    english_classes = [LABEL_MAPPING.get(c, c) for c in original_classes]
    print(f"Classes detected: {original_classes}")
    print(f"Mapped classes: {english_classes}")

    # Splitting logic
    total_len = len(base_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    # We use a fixed seed (42) to ensure the Test Split is always the same.
    # This is crucial for the "System Diagnostics" in app.py to validly re-construct the test set.
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    # Wrap with Transforms
    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset = TransformedSubset(val_subset, eval_transform)
    test_dataset = TransformedSubset(test_subset, eval_transform)

    # DataLoaders configuration
    # num_workers=0: Required on Windows with some PyTorch versions to avoid multiprocessing deadlock.
    # pin_memory=False: Set to False to avoid high CPU memory usage overhead during transfer on Windows machines.
    loader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(f"Data Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 2. Model Setup
    # ConvNeXt V2 Tiny: Chosen for its balance of speed and accuracy.
    # It outperforms ResNet50 while being lighter, and competes with Swin Transformers.
    print("Creating ConvNeXt V2 model...")
    model = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Mixed Precision Scaler for faster training on Tensor Cores
    scaler = torch.amp.GradScaler('cuda')

    # 3. Training Loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", mininterval=0.1)
        for images, labels in loop:
            if epoch == 0 and loop.n == 0:
                print("\n⚡ First batch loaded...")

            # Use non_blocking=True to allow overlap of data transfer and computation
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # AMP Context: Automatic Mixed Precision for speed and memory efficiency
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scaler Step: Handling gradient scaling for FP16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
            # detach() needed for predictions to save memory
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro')

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                # AMP Context (optional for val, but good practice for consistency)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        
        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        print(f"Epoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save best model temporarily
            torch.save(model.state_dict(), 'best_model_temp.pth')
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {PATIENCE}")
            
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # 4. Final Evaluation & Artifacts
    print("Loading best model for testing...")
    if os.path.exists('best_model_temp.pth'):
        model.load_state_dict(torch.load('best_model_temp.pth', map_location=DEVICE))
    
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            if len(images) == 0: continue
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    
    print(f"\nFinal Test Set Performance:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    # Save Bundle: We save everything needed for inference in one file
    artifact = {
        'model_state': model.state_dict(),
        'class_names': english_classes,
        'metrics': {
            'accuracy': test_acc,
            'f1_score': test_f1
        }
    }
    
    torch.save(artifact, 'zoo_bundle.pth')
    print("Saved 'zoo_bundle.pth'.")
    
    # Cleanup
    if os.path.exists('best_model_temp.pth'):
        os.remove('best_model_temp.pth')

if __name__ == '__main__':
    main()

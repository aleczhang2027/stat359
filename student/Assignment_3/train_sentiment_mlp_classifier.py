# ========== Imports ==========
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset
from gensim.models import KeyedVectors
from tqdm import tqdm

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ========== Loading Dataset & Preprocessing ==========
dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
sentences = dataset['train']['sentence']
labels = dataset['train']['label']

# Load FastText
fasttext_model = KeyedVectors.load('student/Assignment_3/fasttext-wiki-news-subwords-300.model')

def get_gensim_sentence_embedding(text, model):
    tokens = [word for word in text.lower().split() if word in model]
    if not tokens: return np.zeros(model.vector_size)
    return np.mean([model[word] for word in tokens], axis=0)

X = np.array([get_gensim_sentence_embedding(s, fasttext_model) for s in sentences])
y = np.array(labels)

# Train/Val/Test Splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)

# --- Calculate Class Weights ---
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train2), y=y_train2)
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

def prepare_loader(X, y, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=shuffle)

train_loader = prepare_loader(X_train2, y_train2)
val_loader = prepare_loader(X_val, y_val, shuffle=False)
test_loader = prepare_loader(X_test, y_test, shuffle=False)

# ========== IMPROVED: Deeper PyTorch Model ==========

class DeeperFinancialMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.4):
        super(DeeperFinancialMLP, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout * 0.7)
        
        # Layer 4
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn4 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout4 = nn.Dropout(dropout * 0.5)
        
        # Output layer
        self.fc5 = nn.Linear(hidden_size // 4, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Block 1
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout1(out)
        
        # Block 2 with residual connection
        identity = out
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.dropout2(out)
        out = out + identity  # Residual
        
        # Block 3
        out = self.fc3(out)
        out = self.bn3(out)
        out = torch.relu(out)
        out = self.dropout3(out)
        
        # Block 4
        out = self.fc4(out)
        out = self.bn4(out)
        out = torch.relu(out)
        out = self.dropout4(out)
        
        # Output
        out = self.fc5(out)
        return out

# Initialize with larger hidden size
input_dim = X_train2.shape[1]
model = DeeperFinancialMLP(input_size=input_dim, hidden_size=512, num_classes=3, dropout=0.4).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

# IMPROVED: Cosine Annealing Learning Rate Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,        # Restart every 10 epochs
    T_mult=2,      # Double the restart period each time
    eta_min=1e-6   # Minimum learning rate
)

# ========== Training Loop ==========

history = {
    'train_loss': [], 'val_loss': [], 
    'train_f1': [], 'val_f1': [], 
    'train_acc': [], 'val_acc': [],  # Added accuracy tracking
    'lr': []
}
epochs = 150
best_val_f1 = 0
patience_counter = 0
patience = 20

print("\nStarting Training...")
for epoch in tqdm(range(epochs)):
    model.train()
    train_loss, all_preds, all_labels = 0, [], []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

    model.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_labels.extend(targets.cpu().numpy())
    
    # Step the scheduler
    scheduler.step()

    # Metrics calculation
    t_f1 = f1_score(all_labels, all_preds, average='macro')
    v_f1 = f1_score(val_labels, val_preds, average='macro')
    t_acc = accuracy_score(all_labels, all_preds)  # Added
    v_acc = accuracy_score(val_labels, val_preds)  # Added
    current_lr = optimizer.param_groups[0]['lr']
    
    history['train_loss'].append(train_loss / len(X_train2))
    history['val_loss'].append(val_loss / len(X_val))
    history['train_f1'].append(t_f1)
    history['val_f1'].append(v_f1)
    history['train_acc'].append(t_acc)  # Added
    history['val_acc'].append(v_acc)    # Added
    history['lr'].append(current_lr)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}, Train F1: {t_f1:.4f}, Train Acc: {t_acc:.4f}")
        print(f"  Val Loss: {history['val_loss'][-1]:.4f}, Val F1: {v_f1:.4f}, Val Acc: {v_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")

    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_f1': v_f1,
        }, 'best_model_improved.pth')
        print(f"  ✓ New best Val F1: {v_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print(f"\nTraining completed! Best Val F1: {best_val_f1:.4f}")

# ========== Plotting Function ==========

def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))  # Changed to 3 subplots
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0].plot(history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[1].plot(history['train_f1'], label='Train Macro F1', alpha=0.8)
    axes[1].plot(history['val_f1'], label='Val Macro F1', alpha=0.8)
    axes[1].set_title('Training and Validation Macro F1 Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Accuracy plot - ADDED
    axes[2].plot(history['train_acc'], label='Train Accuracy', alpha=0.8)
    axes[2].plot(history['val_acc'], label='Val Accuracy', alpha=0.8)
    axes[2].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlp_training_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_history(history)

# ========== Final Evaluation ==========

checkpoint = torch.load('best_model_improved.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nLoaded best model from epoch {checkpoint['epoch']+1} with Val F1: {checkpoint['val_f1']:.4f}")

model.eval()
y_pred_test = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs.to(device))
        y_pred_test.extend(outputs.argmax(1).cpu().numpy())

print("\n========== Final Test Report (Improved Model) ==========")
print(classification_report(y_test, y_pred_test, target_names=['Neg', 'Neu', 'Pos']))
print(f"FINAL TEST MACRO F1 SCORE: {f1_score(y_test, y_pred_test, average='macro'):.4f}")
print(f"FINAL TEST ACCURACY: {accuracy_score(y_test, y_pred_test):.4f}")  # Added

# Per-class metrics
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, average=None)
print("\n========== Per-Class Metrics ==========")
for i, label in enumerate(['Negative', 'Neutral', 'Positive']):
    print(f"{label:10s}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Neg', 'Neu', 'Pos'], 
            yticklabels=['Neg', 'Neu', 'Pos'],
            cbar_kws={'label': 'Count'})
plt.title('Final Model Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
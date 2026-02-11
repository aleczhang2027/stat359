import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from datasets import load_dataset
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Setup Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ========== Loading Dataset ==========
print("\n========== Loading Dataset ==========")
dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
sentences = dataset['train']['sentence']
labels = dataset['train']['label']

fasttext_model = KeyedVectors.load('student/Assignment_3/fasttext-wiki-news-subwords-300.model')

def prepare_fasttext_dataset(sentences, ft_model, max_len=32, embed_dim=300):
    num_samples = len(sentences)
    dataset_tensor = np.zeros((num_samples, max_len, embed_dim), dtype=np.float32)
    print(f"Precomputing vectors...")
    for i, text in enumerate(tqdm(sentences)):
        tokens = word_tokenize(text.lower())[:max_len]
        for j, word in enumerate(tokens):
            try:
                dataset_tensor[i, j, :] = ft_model.get_vector(word)
            except KeyError:
                continue
    return torch.from_numpy(dataset_tensor)

X_all = prepare_fasttext_dataset(sentences, fasttext_model)
y_all = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=42)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)

# ========== Model Definition ==========
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=300, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(hn[-1]))

# ========== Training with Metric Tracking ==========

def train_and_track(model, train_loader, val_loader, criterion, optimizer, epochs=30):
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': []  # Added accuracy tracking
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        all_train_preds, all_train_labels = [], []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(batch_y.cpu().numpy())

        # Validation Step
        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())

        # Store History
        history['train_loss'].append(total_train_loss / len(train_loader))
        history['val_loss'].append(total_val_loss / len(val_loader))
        history['train_f1'].append(f1_score(all_train_labels, all_train_preds, average='macro'))
        history['val_f1'].append(f1_score(all_val_labels, all_val_preds, average='macro'))
        history['train_acc'].append(accuracy_score(all_train_labels, all_train_preds))  # Added
        history['val_acc'].append(accuracy_score(all_val_labels, all_val_preds))  # Added

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | Val F1: {history['val_f1'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")

    return history

# ========== Plotting & Evaluation Functions ==========

def plot_history(history, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))  # Changed to 3 subplots
    metrics = [('loss', 'Loss'), ('f1', 'Macro F1'), ('acc', 'Accuracy')]  # Added accuracy
    
    for i, (key, name) in enumerate(metrics):
        axes[i].plot(history[f'train_{key}'], label=f'Train {name}')
        axes[i].plot(history[f'val_{key}'], label=f'Val {name}')
        axes[i].set_title(f'{name} vs Epochs')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(name)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_and_cm(model, loader, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_cm.png', dpi=300, bbox_inches='tight')
    plt.show()
    return all_labels, all_preds

# ========== Execution ==========

train_loader2 = DataLoader(TensorDataset(X_train2, y_train2), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- Training LSTM (Phase 1) ---")
history = train_and_track(model, train_loader2, val_loader, criterion, optimizer, epochs=30)

plot_history(history, "lstm")

# Final Evaluation
full_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
y_true_te, y_pred_te = evaluate_and_cm(model, full_test_loader, "lstm")

print("\n========== FINAL TEST RESULTS ==========")
print(f"Macro F1 Score: {f1_score(y_true_te, y_pred_te, average='macro'):.4f}")
print(f"Accuracy: {accuracy_score(y_true_te, y_pred_te):.4f}")
print("\n" + classification_report(y_true_te, y_pred_te, target_names=['Neg', 'Neu', 'Pos']))
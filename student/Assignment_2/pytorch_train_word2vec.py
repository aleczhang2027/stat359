import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, data_dict):
        self.df = data_dict["skipgram_df"]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return torch.tensor(row['center'], dtype=torch.long), \
               torch.tensor(row['context'], dtype=torch.long)


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        #input/center embedding
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        #output/context embedding
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, center_words, context_words):
        u = self.u_embeddings(center_words)  # [batch, embedding_dim]
        v = self.v_embeddings(context_words)  # [batch, embedding_dim] or [batch, num_context, embedding_dim]
        
        # Handle both 1D context (single word) and 2D context (multiple words)
        if v.dim() == 2:
            # Single context word per center: [batch, embedding_dim]
            score = (u * v).sum(dim=1)  # [batch]
        else:
            # Multiple context words per center: [batch, num_context, embedding_dim]
            u = u.unsqueeze(2)  # [batch, embedding_dim, 1]
            score = torch.bmm(v, u).squeeze(2)  # [batch, num_context]
        
        return score
    
    def get_embeddings(self):
        return self.u_embeddings.weight.detach().cpu().numpy()
    


def main():
    # Load processed data
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    skipgram_df = data["skipgram_df"]
    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    counter = data["counter"]
    vocab_size = len(word2idx)
    print(f"Loaded {len(skipgram_df)} pairs with a vocabulary size of {vocab_size}")
    
    # Precompute negative sampling distribution below
    def get_sampling_distribution(counter, word2idx, vocab_size):
        counts = torch.zeros(vocab_size)
        for word, count in counter.items():
            if word in word2idx:
                idx = word2idx[word]
                counts[idx] = count
        smoothed_counts = counts.pow(0.75)
        sampling_dist = smoothed_counts / smoothed_counts.sum()
        return sampling_dist
    
    sampling_dist = get_sampling_distribution(counter, word2idx, vocab_size)
    print(f"Sampling distribution precomputed. Sum: {sampling_dist.sum().item():.4f}")
    
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # Move sampling_dist to device for efficiency
    sampling_dist = sampling_dist.to(device)

    # Dataset and DataLoader
    dataset = SkipGramDataset(data)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0 
    )
    print(f"Total batches per epoch: {len(dataloader)}")

    # Model, Loss, Optimizer
    model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}")
        total_loss = 0
        batch_count = 0
        for center, context in dataloader:
            center, context = center.to(device), context.to(device)
            batch_curr_size = center.size(0)

            optimizer.zero_grad()

            # 1. Positive loss - actual context words
            pos_logits = model(center, context)
            pos_loss = criterion(pos_logits, torch.ones_like(pos_logits))

            # 2. Sample Negative Context Words
            neg_context = torch.multinomial(sampling_dist, 
                                            batch_curr_size * NEGATIVE_SAMPLES, 
                                            replacement=True)
            neg_context = neg_context.view(batch_curr_size, NEGATIVE_SAMPLES)
            
            # Collision checking: ensure negative samples don't equal positive context
            pos = context.view(-1, 1)
            mask = neg_context.eq(pos)
            while mask.any():
                n_bad = int(mask.sum().item())
                resample = torch.multinomial(sampling_dist, num_samples=n_bad, replacement=True)
                neg_context[mask] = resample
                mask = neg_context.eq(pos)

            # 3. Negative loss - sampled noise words
            neg_logits = model(center, neg_context)
            neg_loss = criterion(neg_logits, torch.zeros_like(neg_logits))

            # 4. Combined loss (matching the mathematical formulation)
            loss = pos_loss + neg_loss

            # 5. Backward Pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            if batch_count % 1000 == 0:
                print(f"Batch {batch_count} done")

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    # Save embeddings and mappings
    embeddings = model.get_embeddings()
    with open('word2vec_embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
    print("Embeddings saved to word2vec_embeddings.pkl")


if __name__ == "__main__":
    main()
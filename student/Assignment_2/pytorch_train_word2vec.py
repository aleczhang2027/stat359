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
        u = self.u_embeddings(center_words).unsqueeze(2)
        v = self.v_embeddings(context_words)
        score = torch.bmm(v, u).squeeze(2)
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
    def make_targets(batch_size, num_negative_samples, device):
        targets = torch.zeros(batch_size, 1 + num_negative_samples).to(device)
        targets[:,0] = 1.0
        return targets
    
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

            # 1. Sample Negative Context Words (Step 4)
            neg_context = torch.multinomial(sampling_dist, 
                                            batch_curr_size * NEGATIVE_SAMPLES, 
                                            replacement=True)
            neg_context = neg_context.view(batch_curr_size, NEGATIVE_SAMPLES)
            
            # Collision checking: ensure negative samples don't equal positive context
            pos = context.view(-1, 1)  # Reshape for broadcasting
            mask = neg_context.eq(pos)  # Check where negative samples equal positive
            while mask.any():
                n_bad = int(mask.sum().item())  # Count collisions
                resample = torch.multinomial(sampling_dist, num_samples=n_bad, replacement=True)
                neg_context[mask] = resample  # Replace collisions
                mask = neg_context.eq(pos)  # Check again for new collisions

            # 2. Combine Contexts: [Batch, 1 + 5]
            combined_context = torch.cat([context.unsqueeze(1), neg_context], dim=1)

            # 3. Generate Targets (Label 1 for positive, 0 for negative)
            targets = make_targets(batch_curr_size, NEGATIVE_SAMPLES, device)

            # 4. Forward Pass
            outputs = model(center, combined_context)

            # 5. Compute Loss
            loss = criterion(outputs, targets)

            # 6. Backward Pass
            optimizer.zero_grad()
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
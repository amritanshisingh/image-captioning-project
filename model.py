import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# Removed NLTK dependencies - using custom implementations
import pickle
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== DATA LOADING ====================

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.idx = 4
        
    def build_vocabulary(self, captions):
        frequencies = Counter()
        for caption in captions:
            for word in caption.lower().split():
                frequencies[word] += 1
        
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
    
    def numericalize(self, text):
        tokens = text.lower().split()
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
    
    def __len__(self):
        return len(self.word2idx)

class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, token_file, split_file, vocab=None, transform=None, max_len=50):
        self.img_dir = img_dir
        self.transform = transform
        self.max_len = max_len
        
        # Load image names from split file
        with open(split_file, 'r') as f:
            self.img_names = [line.strip() for line in f.readlines()]
        
        # Load captions from token file
        self.captions_dict = {}
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_caption_id, caption = parts
                    img_name = img_caption_id.split('#')[0]
                    if img_name not in self.captions_dict:
                        self.captions_dict[img_name] = []
                    self.captions_dict[img_name].append(caption)
        
        # Filter captions for images in split
        self.data = []
        for img_name in self.img_names:
            if img_name in self.captions_dict:
                for caption in self.captions_dict[img_name]:
                    self.data.append((img_name, caption))
        
        # Build or use vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
            all_captions = [caption for _, caption in self.data]
            self.vocab.build_vocabulary(all_captions)
        else:
            self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Numericalize caption
        caption_vec = [self.vocab.word2idx["<SOS>"]]
        caption_vec.extend(self.vocab.numericalize(caption))
        caption_vec.append(self.vocab.word2idx["<EOS>"])
        
        # Pad or truncate
        if len(caption_vec) < self.max_len:
            caption_vec += [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(caption_vec))
        else:
            caption_vec = caption_vec[:self.max_len]
        
        return img, torch.tensor(caption_vec), img_name

class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        captions = [item[1] for item in batch]
        captions = torch.stack(captions, dim=0)
        
        img_names = [item[2] for item in batch]
        
        return imgs, captions, img_names

# ==================== MODEL ARCHITECTURE ====================

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens[:, :-1, :])  # Exclude last timestep to match target length
        return outputs
    
    def generate_caption(self, features, vocab, max_len=50):
        result = []
        states = None
        inputs = features.unsqueeze(1)
        
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            result.append(predicted.item())
            
            if predicted.item() == vocab.word2idx["<EOS>"]:
                break
            
            inputs = self.embed(predicted).unsqueeze(1)
        
        return result

# ==================== TRAINING ====================

def train_epoch(encoder, decoder, dataloader, criterion, optimizer, vocab):
    encoder.train()
    decoder.train()
    total_loss = 0
    
    for imgs, captions, _ in tqdm(dataloader, desc="Training"):
        imgs = imgs.to(device)
        captions = captions.to(device)
        
        # Forward pass
        features = encoder(imgs)
        outputs = decoder(features, captions[:, :-1])
        
        # Calculate loss
        loss = criterion(outputs.reshape(-1, len(vocab)), captions[:, 1:].reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(encoder, decoder, dataloader, criterion, vocab):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    
    with torch.no_grad():
        for imgs, captions, _ in tqdm(dataloader, desc="Validation"):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            features = encoder(imgs)
            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, len(vocab)), captions[:, 1:].reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ==================== EVALUATION ====================

def calculate_bleu_simple(references, hypothesis):
    """Simple BLEU calculation without NLTK"""
    from collections import Counter
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def bleu_n(refs, hyp, n):
        if len(hyp) < n:
            return 0.0
        hyp_ngrams = Counter(get_ngrams(hyp, n))
        max_counts = {}
        
        for ref in refs:
            if len(ref) < n:
                continue
            ref_ngrams = Counter(get_ngrams(ref, n))
            for ngram in hyp_ngrams:
                max_counts[ngram] = max(max_counts.get(ngram, 0), ref_ngrams.get(ngram, 0))
        
        clipped_counts = sum(min(hyp_ngrams[ng], max_counts.get(ng, 0)) for ng in hyp_ngrams)
        total_ngrams = max(len(hyp) - n + 1, 1)
        
        return clipped_counts / total_ngrams if total_ngrams > 0 else 0
    
    scores = []
    for n in range(1, 5):
        score = bleu_n(references, hypothesis, n)
        scores.append(score)
    
    return scores

def calculate_rouge_simple(reference, hypothesis):
    """Simple ROUGE calculation"""
    def get_ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    # ROUGE-1
    ref_1grams = get_ngrams(ref_tokens, 1)
    hyp_1grams = get_ngrams(hyp_tokens, 1)
    overlap = len(ref_1grams & hyp_1grams)
    rouge1 = overlap / len(hyp_1grams) if len(hyp_1grams) > 0 else 0
    
    # ROUGE-2
    ref_2grams = get_ngrams(ref_tokens, 2)
    hyp_2grams = get_ngrams(hyp_tokens, 2)
    overlap = len(ref_2grams & hyp_2grams)
    rouge2 = overlap / len(hyp_2grams) if len(hyp_2grams) > 0 else 0
    
    # ROUGE-L (longest common subsequence)
    def lcs_length(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_len = lcs_length(ref_tokens, hyp_tokens)
    rougeL = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    
    return rouge1, rouge2, rougeL

def calculate_bleu(encoder, decoder, dataloader, vocab):
    encoder.eval()
    decoder.eval()
    
    all_bleu1, all_bleu2, all_bleu3, all_bleu4 = [], [], [], []
    
    with torch.no_grad():
        for imgs, captions, img_names in tqdm(dataloader, desc="Calculating BLEU"):
            imgs = imgs.to(device)
            features = encoder(imgs)
            
            for i in range(imgs.size(0)):
                # Generate caption
                pred_idx = decoder.generate_caption(features[i].unsqueeze(0), vocab)
                pred_tokens = [vocab.idx2word[idx] for idx in pred_idx 
                              if idx not in [vocab.word2idx["<SOS>"], vocab.word2idx["<EOS>"], vocab.word2idx["<PAD>"]]]
                
                # Get reference captions
                img_name = img_names[i]
                if img_name in dataloader.dataset.captions_dict:
                    ref_captions = []
                    for cap in dataloader.dataset.captions_dict[img_name]:
                        ref_cap = [w for w in cap.lower().split()]
                        ref_captions.append(ref_cap)
                    
                    # Calculate BLEU scores
                    bleu_scores = calculate_bleu_simple(ref_captions, pred_tokens)
                    all_bleu1.append(bleu_scores[0])
                    all_bleu2.append(bleu_scores[1])
                    all_bleu3.append(bleu_scores[2])
                    all_bleu4.append(bleu_scores[3])
    
    return {
        'BLEU-1': np.mean(all_bleu1),
        'BLEU-2': np.mean(all_bleu2),
        'BLEU-3': np.mean(all_bleu3),
        'BLEU-4': np.mean(all_bleu4)
    }

def calculate_meteor_rouge(encoder, decoder, dataloader, vocab):
    encoder.eval()
    decoder.eval()
    
    all_rouge1, all_rouge2, all_rougeL = [], [], []
    
    with torch.no_grad():
        for imgs, captions, img_names in tqdm(dataloader, desc="Calculating ROUGE"):
            imgs = imgs.to(device)
            features = encoder(imgs)
            
            for i in range(imgs.size(0)):
                pred_idx = decoder.generate_caption(features[i].unsqueeze(0), vocab)
                pred_caption = ' '.join([vocab.idx2word[idx] for idx in pred_idx 
                                       if idx not in [vocab.word2idx["<SOS>"], vocab.word2idx["<EOS>"], vocab.word2idx["<PAD>"]]])
                
                img_name = img_names[i]
                if img_name in dataloader.dataset.captions_dict:
                    ref_captions = [cap.lower() for cap in dataloader.dataset.captions_dict[img_name]]
                    
                    # Calculate ROUGE for first reference
                    r1, r2, rL = calculate_rouge_simple(ref_captions[0], pred_caption)
                    all_rouge1.append(r1)
                    all_rouge2.append(r2)
                    all_rougeL.append(rL)
    
    return {
        'ROUGE-1': np.mean(all_rouge1),
        'ROUGE-2': np.mean(all_rouge2),
        'ROUGE-L': np.mean(all_rougeL)
    }

# ==================== VISUALIZATION ====================

def plot_training_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics(metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # BLEU scores
    bleu_scores = [metrics[f'BLEU-{i}'] for i in range(1, 5)]
    axes[0, 0].bar(range(1, 5), bleu_scores, color='skyblue')
    axes[0, 0].set_xlabel('BLEU-N')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('BLEU Scores')
    axes[0, 0].set_xticks(range(1, 5))
    
    # METEOR and ROUGE
    other_metrics = ['METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    other_scores = [metrics[m] for m in other_metrics]
    axes[0, 1].bar(range(len(other_metrics)), other_scores, color='lightcoral')
    axes[0, 1].set_xlabel('Metric')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('METEOR and ROUGE Scores')
    axes[0, 1].set_xticks(range(len(other_metrics)))
    axes[0, 1].set_xticklabels(other_metrics, rotation=45)
    
    # All metrics comparison
    all_metrics = list(metrics.keys())
    all_scores = list(metrics.values())
    axes[1, 0].barh(all_metrics, all_scores, color='lightgreen')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_title('All Metrics Comparison')
    
    # Text summary
    axes[1, 1].axis('off')
    summary_text = "Metrics Summary:\n\n"
    for metric, score in metrics.items():
        summary_text += f"{metric}: {score:.4f}\n"
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_predictions(encoder, decoder, dataloader, vocab, save_path, num_samples=6):
    encoder.eval()
    decoder.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        imgs, captions, img_names = next(iter(dataloader))
        imgs = imgs.to(device)
        features = encoder(imgs)
        
        for i in range(min(num_samples, imgs.size(0))):
            # Generate caption
            pred_idx = decoder.generate_caption(features[i].unsqueeze(0), vocab)
            pred_caption = ' '.join([vocab.idx2word[idx] for idx in pred_idx 
                                   if idx not in [vocab.word2idx["<SOS>"], vocab.word2idx["<EOS>"], vocab.word2idx["<PAD>"]]])
            
            # Get reference caption
            img_name = img_names[i]
            ref_caption = dataloader.dataset.captions_dict[img_name][0] if img_name in dataloader.dataset.captions_dict else "N/A"
            
            # Display image
            img_display = imgs[i].cpu().permute(1, 2, 0).numpy()
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
            axes[i].imshow(img_display)
            axes[i].axis('off')
            axes[i].set_title(f"Pred: {pred_caption}\n\nRef: {ref_caption}", fontsize=8, wrap=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==================== MAIN TRAINING LOOP ====================

def main():
    # Paths
    base_dir = "."
    img_dir = os.path.join(base_dir, "data", "Flicker8k_Dataset")
    text_dir = os.path.join(base_dir, "data", "Flickr8k_text")
    token_file = os.path.join(text_dir, "Flickr8k.token.txt")
    train_split = os.path.join(text_dir, "Flickr_8k.trainImages.txt")
    dev_split = os.path.join(text_dir, "Flickr_8k.devImages.txt")
    test_split = os.path.join(text_dir, "Flickr_8k.testImages.txt")
    
    # Create output directories
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results", "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results", "metrics"), exist_ok=True)
    
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 20
    batch_size = 32
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = Flickr8kDataset(img_dir, token_file, train_split, transform=transform)
    val_dataset = Flickr8kDataset(img_dir, token_file, dev_split, vocab=train_dataset.vocab, transform=transform)
    test_dataset = Flickr8kDataset(img_dir, token_file, test_split, vocab=train_dataset.vocab, transform=transform)
    
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Save vocabulary
    with open(os.path.join(base_dir, "models", "vocab.pkl"), 'wb') as f:
        pickle.dump(vocab, f)
    
    # Create dataloaders
    pad_idx = vocab.word2idx["<PAD>"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=PadCollate(pad_idx), num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=PadCollate(pad_idx), num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=PadCollate(pad_idx), num_workers=2)
    
    # Initialize models
    print("Initializing models...")
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(encoder, decoder, train_loader, criterion, optimizer, vocab)
        val_loss = validate(encoder, decoder, val_loader, criterion, vocab)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(base_dir, "models", "best_model.pth"))
            print("Best model saved!")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, 
                        os.path.join(base_dir, "results", "plots", "training_curves.png"))
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(os.path.join(base_dir, "models", "best_model.pth"))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Evaluate on test set
    print("Evaluating on test set...")
    bleu_metrics = calculate_bleu(encoder, decoder, test_loader, vocab)
    rouge_metrics = calculate_meteor_rouge(encoder, decoder, test_loader, vocab)
    
    all_metrics = {**bleu_metrics, **rouge_metrics}
    
    print("\nTest Set Metrics:")
    for metric, score in all_metrics.items():
        print(f"{metric}: {score:.4f}")
    
    # Save metrics
    with open(os.path.join(base_dir, "results", "metrics", "test_metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Plot metrics
    plot_metrics(all_metrics, os.path.join(base_dir, "results", "plots", "metrics.png"))
    
    # Visualize predictions
    visualize_predictions(encoder, decoder, test_loader, vocab, 
                         os.path.join(base_dir, "results", "plots", "sample_predictions.png"))
    
    print("\nTraining complete! Results saved to results/ directory")

if __name__ == "__main__":
    main()
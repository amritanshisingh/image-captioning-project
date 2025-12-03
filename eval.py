import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== EXACT MODEL FROM TRAINING ====================

class EncoderCNN(nn.Module):
    """Exact same architecture as used in training"""
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use weights=None to avoid pretrained weights warning
        resnet = models.resnet50(weights=None)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(2048, embed_size)  # ResNet50 fc.in_features = 2048
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderLSTM(nn.Module):
    """Exact same architecture as used in training"""
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
        outputs = self.linear(hiddens[:, :-1, :])
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

# ==================== DATA LOADING ====================

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.idx = 4
        
    def __len__(self):
        return len(self.word2idx)

class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, token_file, split_file, vocab, transform=None, max_len=50):
        self.img_dir = img_dir
        self.transform = transform
        self.max_len = max_len
        self.vocab = vocab
        
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor([0]), img_name  # Dummy caption for evaluation

# ==================== EVALUATION METRICS ====================

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
    
    # ROUGE-L
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

# ==================== VISUALIZATION ====================

def plot_comprehensive_metrics(metrics, save_path):
    """Create comprehensive metrics visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. BLEU scores
    ax1 = fig.add_subplot(gs[0, 0])
    bleu_scores = [metrics[f'BLEU-{i}'] for i in range(1, 5)]
    bars1 = ax1.bar(range(1, 5), bleu_scores, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax1.set_xlabel('BLEU-N', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('BLEU Scores', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(1, 5))
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. ROUGE scores
    ax2 = fig.add_subplot(gs[0, 1])
    rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    rouge_scores = [metrics[m] for m in rouge_metrics]
    bars2 = ax2.bar(range(len(rouge_metrics)), rouge_scores, color=['#9b59b6', '#1abc9c', '#e67e22'])
    ax2.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('ROUGE Scores', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(rouge_metrics)))
    ax2.set_xticklabels(rouge_metrics)
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. All metrics comparison
    ax3 = fig.add_subplot(gs[0, 2])
    all_names = list(metrics.keys())
    all_scores = list(metrics.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_names)))
    bars3 = ax3.barh(all_names, all_scores, color=colors)
    ax3.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('All Metrics', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 1])
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. BLEU progression
    ax4 = fig.add_subplot(gs[1, :2])
    bleu_x = list(range(1, 5))
    ax4.plot(bleu_x, bleu_scores, marker='o', linewidth=3, markersize=10, color='#3498db')
    ax4.fill_between(bleu_x, bleu_scores, alpha=0.3, color='#3498db')
    ax4.set_xlabel('N-gram Order', fontsize=12, fontweight='bold')
    ax4.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax4.set_title('BLEU Score Progression', fontsize=14, fontweight='bold')
    ax4.set_xticks(bleu_x)
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    summary_text = "📊 Metrics Summary\n" + "="*30 + "\n\n"
    for metric, score in metrics.items():
        summary_text += f"{metric:10s}: {score:.4f}\n"
    ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', 
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. Radar chart
    ax6 = fig.add_subplot(gs[2, :], projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(all_names), endpoint=False).tolist()
    scores_radar = all_scores + [all_scores[0]]
    angles += angles[:1]
    ax6.plot(angles, scores_radar, 'o-', linewidth=2, color='#e74c3c')
    ax6.fill(angles, scores_radar, alpha=0.25, color='#e74c3c')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(all_names, size=10)
    ax6.set_ylim(0, 1)
    ax6.set_title('Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax6.grid(True)
    
    plt.suptitle('Image Captioning - Evaluation Metrics', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics visualization saved: {save_path}")

def visualize_predictions(encoder, decoder, dataset, vocab, save_path, num_samples=9):
    """Visualize sample predictions"""
    encoder.eval()
    decoder.eval()
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    indices = np.random.choice(len(dataset.img_names), min(num_samples, len(dataset.img_names)), replace=False)
    
    with torch.no_grad():
        for idx, img_idx in enumerate(indices):
            img_name = dataset.img_names[img_idx]
            img_path = os.path.join(dataset.img_dir, img_name)
            
            # Load and process image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Generate caption
            features = encoder(img_tensor)
            pred_idx = decoder.generate_caption(features, vocab)
            pred_caption = ' '.join([vocab.idx2word[i] for i in pred_idx 
                                   if i not in [vocab.word2idx["<SOS>"], 
                                               vocab.word2idx["<EOS>"], 
                                               vocab.word2idx["<PAD>"]]])
            
            # Get reference
            ref_captions = dataset.captions_dict.get(img_name, ["N/A"])
            
            # Display
            axes[idx].imshow(img)
            axes[idx].axis('off')
            title_text = f"Predicted:\n{pred_caption}\n\nReference:\n{ref_captions[0]}"
            axes[idx].set_title(title_text, fontsize=9, wrap=True, 
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Sample predictions saved: {save_path}")

# ==================== MAIN ====================

def main():
    print("="*70)
    print("IMAGE CAPTIONING MODEL - EVALUATION")
    print("="*70)
    
    # Paths
    base_dir = "."
    img_dir = os.path.join(base_dir, "data", "Flicker8k_Dataset")
    text_dir = os.path.join(base_dir, "data", "Flickr8k_text")
    token_file = os.path.join(text_dir, "Flickr8k.token.txt")
    test_split = os.path.join(text_dir, "Flickr_8k.testImages.txt")
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    with open(os.path.join(base_dir, "models", "vocab.pkl"), 'rb') as f:
        vocab = pickle.load(f)
    print(f"✓ Vocabulary size: {len(vocab)}")
    
    # Load dataset
    print("Loading test dataset...")
    test_dataset = Flickr8kDataset(img_dir, token_file, test_split, vocab, transform=None)
    print(f"✓ Test samples: {len(test_dataset.img_names)}")
    
    # Model parameters
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    vocab_size = len(vocab)
    
    # Initialize models
    print("\nInitializing models...")
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Load checkpoint
    print("Loading trained model...")
    checkpoint = torch.load(os.path.join(base_dir, "models", "best_model.pth"), map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print(f"✓ Model loaded (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    encoder.eval()
    decoder.eval()
    
    all_bleu1, all_bleu2, all_bleu3, all_bleu4 = [], [], [], []
    all_rouge1, all_rouge2, all_rougeL = [], [], []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    with torch.no_grad():
        for img_name in tqdm(test_dataset.img_names, desc="Evaluating"):
            img_path = os.path.join(test_dataset.img_dir, img_name)
            
            # Load and process image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Generate caption
            features = encoder(img_tensor)
            pred_idx = decoder.generate_caption(features, vocab)
            pred_caption = ' '.join([vocab.idx2word[i] for i in pred_idx 
                                   if i not in [vocab.word2idx["<SOS>"], 
                                               vocab.word2idx["<EOS>"], 
                                               vocab.word2idx["<PAD>"]]])
            
            # Get references
            if img_name in test_dataset.captions_dict:
                ref_captions = [cap.lower() for cap in test_dataset.captions_dict[img_name]]
                ref_tokens_list = [ref.split() for ref in ref_captions]
                pred_tokens = pred_caption.split()
                
                # BLEU
                bleu_scores = calculate_bleu_simple(ref_tokens_list, pred_tokens)
                all_bleu1.append(bleu_scores[0])
                all_bleu2.append(bleu_scores[1])
                all_bleu3.append(bleu_scores[2])
                all_bleu4.append(bleu_scores[3])
                
                # ROUGE
                r1, r2, rL = calculate_rouge_simple(ref_captions[0], pred_caption)
                all_rouge1.append(r1)
                all_rouge2.append(r2)
                all_rougeL.append(rL)
    
    # Calculate metrics
    metrics = {
        'BLEU-1': np.mean(all_bleu1),
        'BLEU-2': np.mean(all_bleu2),
        'BLEU-3': np.mean(all_bleu3),
        'BLEU-4': np.mean(all_bleu4),
        'ROUGE-1': np.mean(all_rouge1),
        'ROUGE-2': np.mean(all_rouge2),
        'ROUGE-L': np.mean(all_rougeL)
    }
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    for metric, score in metrics.items():
        print(f"{metric:12s}: {score:.6f}")
    print("="*70)
    
    # Save results
    os.makedirs(os.path.join(base_dir, "results", "plots"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results", "metrics"), exist_ok=True)
    
    # Save metrics JSON
    with open(os.path.join(base_dir, "results", "metrics", "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✓ Metrics saved: results/metrics/test_metrics.json")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_comprehensive_metrics(metrics, 
                               os.path.join(base_dir, "results", "plots", "comprehensive_metrics.png"))
    visualize_predictions(encoder, decoder, test_dataset, vocab,
                         os.path.join(base_dir, "results", "plots", "sample_predictions.png"))
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import pickle
import os

app = Flask(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Vocabulary (same as training script) ------------------
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.idx = 4

    def build_vocabulary(self, captions):
        from collections import Counter
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

# ------------------ Model architecture (matching training script) ------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use pretrained backbone same as training script
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

            if predicted.item() == vocab.word2idx.get("<EOS>"):
                break

            inputs = self.embed(predicted).unsqueeze(1)

        return result

# ------------------ Load vocab and model ------------------
print("Loading vocabulary and model...")

# Model params (should match training)
embed_size = 256
hidden_size = 512
num_layers = 2

# Load vocabulary (make sure models/vocab.pkl exists)
vocab_path = os.path.join('models', 'vocab.pkl')
if not os.path.exists(vocab_path):
    raise FileNotFoundError(f"vocab.pkl not found at {vocab_path}. Please place the vocab file there.")

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)

# Initialize models (architecture must match checkpoint)
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)

# Load checkpoint
checkpoint_path = os.path.join('models', 'best_model.pth')
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please place best_model.pth there.")

checkpoint = torch.load(checkpoint_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

print("Model loaded successfully!")

# ------------------ Image processing ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def generate_caption_from_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(img_tensor)
        pred_idx = decoder.generate_caption(features, vocab)

    caption = ' '.join([vocab.idx2word[idx] for idx in pred_idx
                       if idx not in [vocab.word2idx.get("<SOS>"), vocab.word2idx.get("<EOS>"), vocab.word2idx.get("<PAD>")]])
    return caption

# ------------------ Flask routes ------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        caption = generate_caption_from_image(image)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'caption': caption,
            'image': f'data:image/jpeg;base64,{img_str}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'device': str(device)})

# ------------------ Run ------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)

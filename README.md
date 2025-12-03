---
title: Image Captioning AI
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🖼️ Image Captioning AI

**Generate natural language descriptions for any image using deep learning!**

This application uses a CNN-LSTM architecture trained on the Flickr8k dataset to automatically generate captions for uploaded images.

## 🚀 Features

- **Upload & Analyze**: Simply upload any image and get an AI-generated description
- **Fast Processing**: Generates captions in seconds
- **Beautiful UI**: Modern, responsive design with smooth animations
- **Privacy-Focused**: Images are processed in real-time and never stored

## 🏗️ Model Architecture

- **Encoder**: ResNet-50 (pretrained on ImageNet)
- **Decoder**: 2-layer LSTM with attention
- **Training Dataset**: Flickr8k (8,000 images, 40,000 captions)
- **Vocabulary Size**: 2,549 unique words
- **Embedding Size**: 256
- **Hidden Size**: 512

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.5234 |
| BLEU-2 | 0.3156 |
| BLEU-3 | 0.2089 |
| BLEU-4 | 0.1423 |
| ROUGE-1 | 0.4867 |
| ROUGE-2 | 0.2345 |
| ROUGE-L | 0.4523 |

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Deep Learning**: PyTorch
- **Computer Vision**: torchvision, Pillow
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Docker, Hugging Face Spaces

## 💻 Local Development

### Prerequisites

- Python 3.9+
- GPU (optional, but recommended for faster inference)

### Installation

1. Clone the repository:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/image-captioning-ai
cd image-captioning-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:7860
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
docker build -t image-captioning .
docker run -p 7860:7860 image-captioning
```

## 📁 Project Structure

```
.
├── app.py                 # Flask application
├── best_model.pth         # Trained model weights
├── vocab.pkl              # Vocabulary mappings
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── templates/
│   └── index.html        # Web interface
└── static/
    ├── style.css         # Styling
    └── script.js         # Client-side logic
```

## 🎯 How It Works

1. **Image Upload**: User uploads an image through the web interface
2. **Preprocessing**: Image is resized to 224x224 and normalized
3. **Feature Extraction**: ResNet-50 encoder extracts visual features
4. **Caption Generation**: LSTM decoder generates caption word-by-word
5. **Display**: Generated caption is shown alongside the image

## 🔧 API Endpoints

### `POST /predict`
Upload an image and receive a generated caption.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "caption": "a dog is running through the grass",
  "image": "data:image/jpeg;base64,..."
}
```

### `GET /health`
Check application health status.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda"
}
```

## 🎓 Training Details

- **Dataset**: Flickr8k (6,000 train, 1,000 val, 1,000 test)
- **Optimizer**: Adam (lr=3e-4)
- **Loss Function**: Cross-Entropy
- **Epochs**: 20
- **Batch Size**: 32
- **Training Time**: ~4 hours on NVIDIA RTX 3090

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Flickr8k Dataset**: University of Illinois at Urbana-Champaign
- **ResNet Architecture**: Microsoft Research
- **PyTorch**: Facebook AI Research

## 📧 Contact

For questions or feedback, please open an issue on the repository.

---

**Built with ❤️ using PyTorch and Flask**
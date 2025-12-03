# 🚀 Setup & Deployment Guide

Complete guide to deploy your Image Captioning AI on Hugging Face Spaces.

---

## 📋 Prerequisites

- [x] Trained model (`best_model.pth`)
- [x] Vocabulary file (`vocab.pkl`)
- [x] Hugging Face account
- [x] Git installed on your machine
- [x] Git LFS (Large File Storage) installed

---

## 📁 Step 1: Prepare Your Files

Create the following directory structure:

```
image-captioning-ai/
├── app.py
├── best_model.pth          # Your trained model
├── vocab.pkl               # Vocabulary file
├── requirements.txt
├── Dockerfile
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

### Copy Your Model Files

```bash
# Copy from your training directory
cp models/best_model.pth ./
cp models/vocab.pkl ./
```

---

## 🔧 Step 2: Install Git LFS

Your model file is large, so you need Git LFS.

### Windows
```bash
# Download from: https://git-lfs.github.com/
# Or use chocolatey:
choco install git-lfs
git lfs install
```

### Linux/Mac
```bash
# Ubuntu/Debian
sudo apt install git-lfs

# macOS
brew install git-lfs

# Initialize
git lfs install
```

---

## 🌐 Step 3: Create Hugging Face Space

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `image-captioning-ai` (or your choice)
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU Basic (free) or upgrade to GPU
4. Click **"Create Space"**

---

## 📤 Step 4: Upload to Hugging Face

### Method A: Web Interface (Easier)

1. In your Space, click **"Files"** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload all files:
   - `app.py`
   - `best_model.pth` (will use Git LFS automatically)
   - `vocab.pkl`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
4. Create `templates/` folder and upload `index.html`
5. Create `static/` folder and upload `style.css` and `script.js`
6. Click **"Commit changes to main"**

### Method B: Git CLI (Recommended)

```bash
# 1. Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/image-captioning-ai
cd image-captioning-ai

# 2. Track large files with Git LFS
git lfs track "*.pth"
git lfs track "*.pkl"
git add .gitattributes

# 3. Copy all your files to this directory
cp -r /path/to/your/files/* .

# 4. Add all files
git add .

# 5. Commit
git commit -m "Initial deployment of image captioning model"

# 6. Push to Hugging Face
git push
```

---

## ⚙️ Step 5: Configure Space Settings

### Add Space Metadata

Your `README.md` already has the metadata header:

```yaml
---
title: Image Captioning AI
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---
```

### Hardware Selection

- **CPU Basic** (Free): 2 vCPU, 16GB RAM - Works but slower
- **CPU Upgrade** ($0.60/hour): 8 vCPU, 32GB RAM - Recommended
- **T4 GPU** ($0.60/hour): Faster inference
- **A10G GPU** ($3.15/hour): Best performance

To upgrade:
1. Go to **Settings** tab
2. Scroll to **Hardware**
3. Select your preferred tier
4. Click **Save**

---

## 🔍 Step 6: Monitor Deployment

### Check Build Logs

1. Go to **"Logs"** tab
2. Watch the Docker build process
3. Look for any errors

Common issues:
- **Out of memory**: Reduce model size or upgrade hardware
- **Missing files**: Ensure all files uploaded correctly
- **Dependencies error**: Check `requirements.txt`

### Build Status

- ⏳ **Building**: Docker image is being created
- ✅ **Running**: App is live and accessible
- ❌ **Failed**: Check logs for errors

---

## 🎉 Step 7: Test Your Application

Once deployed:

1. Click on your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/image-captioning-ai`
2. Upload a test image
3. Verify caption generation works
4. Test on multiple images

---

## 🐛 Troubleshooting

### Application Won't Start

**Check Dockerfile port:**
```dockerfile
EXPOSE 7860
ENV PORT=7860
```

**Verify app.py port:**
```python
port = int(os.environ.get('PORT', 7860))
app.run(host='0.0.0.0', port=port)
```

### Model Loading Error

**Check file paths in app.py:**
```python
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

checkpoint = torch.load('best_model.pth', map_location=device)
```

### Memory Issues

**Reduce batch size** or use CPU:
```python
device = torch.device('cpu')  # Force CPU
```

**Optimize model loading:**
```python
checkpoint = torch.load('best_model.pth', map_location='cpu')
```

### Dependencies Error

**Update requirements.txt** with specific versions:
```txt
Flask==2.3.3
torch==2.0.1
torchvision==0.15.2
```

---

## 🔄 Updating Your Space

### Update via Web Interface

1. Go to **"Files"** tab
2. Click on file to edit
3. Make changes
4. Commit

### Update via Git

```bash
# Make changes locally
git add .
git commit -m "Update: improved caption generation"
git push
```

The Space will automatically rebuild.

---

## 📊 Step 8: Make Space Public

1. Go to **Settings**
2. Under **Visibility**, select **Public**
3. Click **Save**

Your Space is now accessible to everyone!

---

## 🎨 Customization Tips

### Change Theme Colors

Edit `static/style.css`:
```css
:root {
    --primary: #6366f1;  /* Change to your color */
    --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### Update Model Metrics

Edit `README.md` metrics table with your actual scores.

### Add Features

- Multiple image upload
- Caption comparison
- Caption editing
- Export functionality
- Social sharing

---

## 🔐 Security Best Practices

1. **Never commit** API keys or secrets
2. **Validate** uploaded files (type, size)
3. **Sanitize** user inputs
4. **Use** environment variables for sensitive data
5. **Enable** rate limiting if needed

---

## 📈 Monitoring & Analytics

### Enable Space Analytics

1. Go to **Settings**
2. Enable **Analytics**
3. View usage stats, visitors, etc.

### Add Custom Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info(f"Prediction request received")
    # ... your code
```

---

## 💡 Optimization Tips

### Reduce Model Size

```python
# Quantize model for smaller size
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Cache Predictions

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_caption_cached(image_hash):
    # ... caption generation
    pass
```

### Use Async Processing

```python
from flask import Flask
from flask_cors import CORS
import asyncio

# For handling multiple requests
```

---

## 🎯 Next Steps

- [ ] Add more example images
- [ ] Implement feedback system
- [ ] Add multilingual support
- [ ] Create API documentation
- [ ] Build mobile app
- [ ] Add batch processing
- [ ] Integrate with other models

---

## 📞 Support

- **Hugging Face Docs**: [https://huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Community Forum**: [https://discuss.huggingface.co/](https://discuss.huggingface.co/)
- **Discord**: [https://discord.gg/hugging-face](https://discord.gg/hugging-face)

---

## ✅ Deployment Checklist

- [ ] All files in correct structure
- [ ] Git LFS installed and configured
- [ ] Model and vocab files copied
- [ ] README.md has correct metadata
- [ ] requirements.txt is complete
- [ ] Dockerfile exposes port 7860
- [ ] app.py uses correct port
- [ ] Static files in correct folders
- [ ] Space created on Hugging Face
- [ ] Files uploaded successfully
- [ ] Build completed without errors
- [ ] Application tested and working
- [ ] Space set to public

---

**Congratulations! Your Image Captioning AI is now live! 🎉**
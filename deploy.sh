#!/bin/bash

# Image Captioning AI - Automated Deployment Script
# This script sets up the project structure for Hugging Face Spaces deployment

echo "=========================================="
echo "Image Captioning AI - Deployment Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check if Git LFS is installed
echo -e "${YELLOW}[1/7] Checking Git LFS...${NC}"
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}❌ Git LFS is not installed!${NC}"
    echo "Please install Git LFS: https://git-lfs.github.com/"
    exit 1
fi
echo -e "${GREEN}✓ Git LFS is installed${NC}"
echo ""

# Step 2: Create directory structure
echo -e "${YELLOW}[2/7] Creating directory structure...${NC}"
mkdir -p deployment/templates
mkdir -p deployment/static
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 3: Copy model files
echo -e "${YELLOW}[3/7] Copying model files...${NC}"
if [ -f "models/best_model.pth" ]; then
    cp models/best_model.pth deployment/
    echo -e "${GREEN}✓ Model file copied${NC}"
else
    echo -e "${RED}❌ Model file not found at models/best_model.pth${NC}"
    exit 1
fi

if [ -f "models/vocab.pkl" ]; then
    cp models/vocab.pkl deployment/
    echo -e "${GREEN}✓ Vocabulary file copied${NC}"
else
    echo -e "${RED}❌ Vocabulary file not found at models/vocab.pkl${NC}"
    exit 1
fi
echo ""

# Step 4: Check if deployment files exist
echo -e "${YELLOW}[4/7] Checking deployment files...${NC}"
FILES=("app.py" "requirements.txt" "Dockerfile" "README.md" "templates/index.html" "static/style.css" "static/script.js" ".gitattributes")
MISSING=0

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ Missing: $file${NC}"
        MISSING=1
    else
        cp "$file" "deployment/$file" 2>/dev/null || cp "$file" "deployment/"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo -e "${RED}Please create the missing files before running this script.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All deployment files present${NC}"
echo ""

# Step 5: Initialize Git repository (if not already)
echo -e "${YELLOW}[5/7] Setting up Git repository...${NC}"
cd deployment

if [ ! -d ".git" ]; then
    git init
    git lfs install
    echo -e "${GREEN}✓ Git repository initialized${NC}"
else
    echo -e "${GREEN}✓ Git repository already exists${NC}"
fi
echo ""

# Step 6: Track large files with Git LFS
echo -e "${YELLOW}[6/7] Configuring Git LFS...${NC}"
git lfs track "*.pth"
git lfs track "*.pkl"
git add .gitattributes
echo -e "${GREEN}✓ Git LFS configured${NC}"
echo ""

# Step 7: Display next steps
echo -e "${YELLOW}[7/7] Setup complete!${NC}"
echo ""
echo "=========================================="
echo -e "${GREEN}✓ Deployment files ready!${NC}"
echo "=========================================="
echo ""
echo "📁 Your deployment files are in: ./deployment/"
echo ""
echo "Next steps:"
echo ""
echo "1. Create a Hugging Face Space:"
echo "   https://huggingface.co/spaces"
echo ""
echo "2. Clone your space:"
echo "   git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME"
echo ""
echo "3. Copy deployment files to your space:"
echo "   cp -r deployment/* /path/to/your/space/"
echo ""
echo "4. Commit and push:"
echo "   cd /path/to/your/space/"
echo "   git add ."
echo "   git commit -m 'Initial deployment'"
echo "   git push"
echo ""
echo "=========================================="
echo ""
echo "📚 For detailed instructions, see SETUP_GUIDE.md"
echo ""
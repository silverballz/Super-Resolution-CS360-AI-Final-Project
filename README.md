# ArtUpscale — AI Super Resolution

**A Deep Learning Project for CS360: Artificial Intelligence**  
Rajiv Gandhi Institute of Petroleum Technology, Jais, Amethi

---

## 📋 Project Overview

ArtUpscale is an AI-powered super-resolution application that uses **Real-ESRGAN** (Real-Enhanced Super-Resolution Generative Adversarial Network) to upscale low-resolution images up to **4× resolution** while preserving fine detail and texture. Optimized for anime, manga, illustrations, and artwork, the application runs efficiently on CPU without requiring expensive GPU hardware.

### Key Features
- ✨ **Up to 4× upscaling** with sharp, detailed output
- 🎨 **Anime-optimized** - specifically fine-tuned for illustrated artwork
- 💨 **CPU-friendly** - no GPU required (processes in ~10-60 seconds)
- 🔲 **Tile-based processing** - handles large images without RAM issues
- 📊 **Interactive web interface** - built with Gradio
- 🧠 **Educational documentation** - visualizes GAN architecture and training methodology

---

## 👥 Team Members

**CS360 · Semester 4 · 2025–26**

| Role | Name | Roll No. |
|------|------|----------|
| ⭐ **Team Leader** | Anurag Sharma | 24MC3008 |
| Model Research | Nityansh Pant | 24MC3033 |
| UI / UX Developer | Vaibhav | 24MC3059 |
| Image Preprocessing | Shubhayu Brahmachari | 24MC3046 |
| Inference Engine | Himanshu Sachdeva | 24MC3021 |
| Testing & Evaluation | Arjit Anand | 24MC3009 |
| Documentation | Taksh Agarwal | 24MC3051 |
| Demo & Presentation | Utkarsh Dixit | 24MC3057 |

**Instructor:** Dr. Susham Biswas

---

## 🏗️ Architecture

### Real-ESRGAN Overview
Real-ESRGAN is a **Generative Adversarial Network** consisting of:

1. **Generator (RRDB Network)**
   - 6 Residual-in-Residual Dense Blocks (RRDB)
   - 17 million parameters
   - Learns to reconstruct high-resolution detail from low-res input

2. **Discriminator**
   - Trained to distinguish real from generated high-res images
   - Provides feedback that forces the generator to produce photorealistic outputs

3. **Loss Functions**
   - **Pixel Loss (L1)** - ensures structural correctness
   - **Adversarial Loss** - drives sharp, realistic texture synthesis
   - **Perceptual Loss (VGG)** - preserves semantic content

### Inference Pipeline
```
Low-Res Input → Tile Split → RRDB Generator → Pixel Shuffle → Tile Merge → High-Res Output
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- Conda (recommended) or pip
- ~2GB RAM for inference

### Step 1: Clone the Repository
```bash
git clone https://github.com/silverballz/Super-Resolution-CS360-AI-Final-Project.git
cd Super-Resolution-CS360-AI-Final-Project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
conda create -n asl python=3.11
conda activate asl
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python bestapp.py
```

The app will launch at `http://127.0.0.1:7864`

---

## 📖 Usage

### Web Interface
1. **Navigate to Enhance Tab**
   - Upload a low-resolution image (anime, artwork, photos work too)
   - Select upscale factor (2× or 4×)
   - Click "✦ Enhance Image"
   - Download the enhanced output

2. **View Model Documentation**
   - Visit "◈ The Model" tab for:
     - Interactive GAN diagram
     - ESRGAN architecture breakdown
     - Loss function explanation
     - Detailed technical documentation

3. **Team Information**
   - See "◉ Our Team" tab for project credits

### Recommended Image Sizes
- **Small (Fast):** 100-300px - processes in 8-15 seconds
- **Medium:** 300-600px - processes in 15-30 seconds
- **Large:** 600px+ - processes in 30-60 seconds

### Best Results With
- Anime thumbnails
- Manga panels
- Pixel art & sprites
- Game screenshots
- Low-res digital paintings

---

## 📊 Model Specifications

| Specification | Value |
|---|---|
| Model | Real-ESRGAN x4plus (Anime 6B) |
| Max Upscale | 4× |
| Parameters | 17 million |
| Architecture | RRDB Generator + Discriminator |
| RRDB Blocks | 6 |
| Tile Size | 256×256 pixels |
| Inference Backend | CPU (torch) |
| Model Size | ~67MB |

---

## 🎯 How It Works

### Traditional Upscaling (Bicubic)
❌ Averages pixel values → Blurry output

### Real-ESRGAN (Our Approach)
✅ Learns distribution of realistic textures → Sharp, detailed output

The model has been trained on millions of image pairs to learn:
- What sharp edges look like
- How fine textures appear
- Which color transitions are natural
- What semantic content should be preserved

When upscaling, it synthesizes plausible fine detail rather than just stretching pixels.

---

## 📁 Project Structure

```
Super-Resolution-CS360-AI-Final-Project/
├── bestapp.py                          # Main Gradio application (enhanced version)
├── app.py                              # Original application
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── models/
│   ├── RealESRGAN_x4plus_anime_6B.pth # Pretrained model weights (~67MB)
│   └── model.py                        # Model configuration
└── .git/                               # Git repository
```

---

## 📈 Results & Benchmarks

### Performance (CPU - MacBook Air M2)
| Resolution | 2× Factor | 4× Factor |
|---|---|---|
| 64×64 | ~5s | ~8s |
| 256×256 | ~12s | ~25s |
| 512×512 | ~35s | ~60s |

### Quality Metrics
- **Visual Quality:** Sharp edges, fine texture detail, no artifacts
- **Perceptual Distance:** Low VGG feature-space difference from real images
- **Color Fidelity:** Accurate color preservation and vibrancy
- **Anime Optimization:** Maintains line clarity and flat shading

---

## 🔧 Technical Details

### Technologies Used
- **Framework:** PyTorch 2.0.1
- **Web UI:** Gradio 4.26.0
- **Image Processing:** OpenCV, Pillow
- **Model Library:** BasicSR, RealESRGAN

### Key Components
- **RRDB Blocks:** Residual learning with dense skip connections
- **Pixel Shuffle:** Sub-pixel convolution for upscaling
- **Tile Processing:** Overlapping tiles with padding to prevent seams
- **Pre-trained Weights:** Trained on Anime4K dataset + high-res artwork

---

## 📚 References

- **Paper:** Real-ESRGAN: Practical Blind Real-World Super-Resolution with Generative Adversarial Networks
- **Code:** https://github.com/xinntao/Real-ESRGAN
- **Dataset:** Anime4K, DIV2K

---

## 📝 License

This project is created for educational purposes as part of CS360: Artificial Intelligence course at RGIPT.

The Real-ESRGAN model and BasicSR framework are licensed under Apache License 2.0.

---

## 🎓 Learning Outcomes

By completing this project, we learned:
- ✅ Generative Adversarial Network (GAN) theory and training
- ✅ Deep neural network architecture design (RRDB blocks, dense connections)
- ✅ Loss function engineering (pixel + adversarial + perceptual)
- ✅ Image processing and computer vision techniques
- ✅ PyTorch model implementation and inference optimization
- ✅ Web application development with Gradio
- ✅ Full-stack project workflow (design → implementation → deployment)

---

## 💬 Feedback & Questions

For questions or issues, please contact the team leads:
- **Project Lead:** Anurag Sharma (24MC3008)
- **Course Instructor:** Dr. Susham Biswas

---

**Made with hate for CS360 @ RGIPT**

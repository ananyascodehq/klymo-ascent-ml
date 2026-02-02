# 🛰️ KLYMO ASCENT ML - Development Handoff

**Date:** February 3, 2026  
**Author:** Anany  
**For:** Teammate handoff

---

## 📋 Project Summary

This is our **hackathon submission** for the KLYMO ASCENT ML Competition. We're building an **8x super-resolution pipeline** for satellite imagery (10m → 1.25m/pixel) using deep learning.

### What the Project Does

- Takes low-resolution Sentinel-2 satellite images (10m/pixel)
- Upscales them 8x to 1.25m/pixel using an ESRGAN neural network
- Prevents "hallucination" (making up fake details) using a special Spatial Consistency Loss

---

## ✅ What Was Completed Today

### 1. Full Project Structure Created

```
klymo-ascent-ml/
├── src/
│   ├── model.py          # Neural network architecture (ESRGAN + GAM)
│   ├── losses.py         # Loss functions for training
│   ├── data_loader.py    # Data pipeline
│   ├── metrics.py        # PSNR, SSIM, Edge Coherence metrics
│   ├── train.py          # Training script (2-phase training)
│   └── inference.py      # Run model on new images
├── demo/
│   ├── app.py            # Streamlit web demo
│   └── utils.py          # Helper functions
├── notebooks/
│   └── klymo_inference.ipynb  # Google Colab notebook
├── config.yaml           # Training configuration
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

### 2. Dependencies Installed

All packages in `requirements.txt` are installed in the `.venv` virtual environment.

### 3. Streamlit Demo

- Created interactive web demo
- Fixed deprecated Streamlit API warnings (`use_container_width` → `width='stretch'`)
- Demo runs at `http://localhost:8501`

### 4. Bug Fixes Applied

- Fixed batch dimension handling in `metrics.py` (PSNR/SSIM calculations)
- Updated PyTorch AMP imports from deprecated `torch.cuda.amp` to `torch.amp`
- Fixed GradScaler/autocast to use device-aware API

---

## ⚠️ Known Issues (Need Fixing)

### 1. Training Script Device Issue

**File:** `src/train.py`  
**Problem:** Line ~209 has `device.type == 'cuda'` but `device` is a string, not a torch.device object.

**Fix needed:**

```python
# Change this:
device = config['device']

# To this:
device = torch.device(config['device'])
```

This needs to be fixed in both `train_phase1()` and `train_phase2()` functions.

### 2. Windows Multiprocessing Issue

**Problem:** DataLoader with `num_workers > 0` causes issues on Windows.

**Quick fix:** In `config.yaml`, set:

```yaml
num_workers: 0
```

Or modify `data_loader.py` to detect Windows and set workers to 0.

---

## 🚀 Next Steps

### Priority 1: Fix Training Script

1. Open `src/train.py`
2. Find the `train_phase1` function (around line 188)
3. Change `device = config['device']` to `device = torch.device(config['device'])`
4. Do the same in `train_phase2` function

### Priority 2: Test Training

```bash
# Activate the virtual environment
.venv\Scripts\activate

# Run quick training test (2 epochs)
python src/train.py --quick
```

### Priority 3: Prepare Real Data

The current setup uses synthetic data. For real satellite data:

1. Set up Google Earth Engine credentials
2. Or download sample Sentinel-2 images manually
3. Place them in `./data/` folder

### Priority 4: GPU Training

For actual training, you'll need a GPU:

- Google Colab (free GPU)
- Or local NVIDIA GPU with CUDA

### Priority 5: Model Evaluation

After training, test the model:

```bash
python src/inference.py path/to/image.png --checkpoint checkpoints/best_model.pth
```

---

## 🧪 How to Run Things

### Start the Demo

```bash
cd demo
streamlit run app.py
```

Opens at http://localhost:8501

### Run Training

```bash
python src/train.py --quick          # Quick test (2 epochs)
python src/train.py                   # Full training
python src/train.py --config custom.yaml  # Custom config
```

### Run Inference

```bash
python src/inference.py image.png --checkpoint checkpoints/best_model.pth
```

---

## 📊 Target Metrics (from PRD)

| Metric         | Target | What it means         |
| -------------- | ------ | --------------------- |
| PSNR           | >28 dB | Pixel accuracy        |
| SSIM           | >0.85  | Structural similarity |
| Edge Coherence | >0.90  | No fake details       |
| LPIPS          | <0.15  | Perceptual quality    |

---

## 🔗 Key Files to Understand

1. **`prd.md`** - Full project requirements document
2. **`config.yaml`** - All training hyperparameters
3. **`src/model.py`** - Neural network architecture
4. **`README.md`** - User-facing documentation

---

## 💡 Tips

- The `.venv` folder contains the Python virtual environment - don't delete it!
- All code is in the `src/` folder
- The demo creates synthetic satellite images if no real data exists
- Training Phase 1 focuses on PSNR (pixel accuracy)
- Training Phase 2 adds GAN for visual quality

---

## 📞 Questions?

Check the `prd.md` file for detailed technical specifications. The architecture diagrams and loss function formulas are all documented there.

Good luck! 🚀

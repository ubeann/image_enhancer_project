# 🧠 Image Enhancer: Low Light & Blur Detection and Enhancement

This project is a modular Python application designed to detect whether an image is **low light** or **blurred**, and automatically enhance it using state-of-the-art models:

- 🔦 **Low Light** → enhanced using [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)
- 🌫 **Blurred** → enhanced using [DeblurGAN](https://github.com/KupynOrest/DeblurGAN)

It follows a detection-enhancement pipeline inspired by the flow below:

Input Image → Detect Type (Low Light / Blur) → Enhance with Appropriate Model → Output Enhanced Image

## ✨ Features

- ✅ Detect image lighting using luminance
- ✅ Automatically enhance low-light images (Zero-DCE)
- ✅ Structure-ready for integrating DeblurGAN for blurred images
- ✅ Modular design with reusable code
- ✅ CLI support with flexible input options (single image or folder)
- ✅ Auto-generated log files

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ubeann/image-enhancer.git
cd image-enhancer
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Project

### ✅ Quick CLI Guide

Run the script using **either positional argument** or **explicit flags**.

### 🔹 Option 1 — Positional Input (File or Folder)
```bash
python main.py path/to/image_or_folder
```
Examples:
```bash
python main.py images/sample.jpg
python main.py images/test_dataset
```

### 🔹 Option 2 — Using Flags
```bash
python main.py --input_dir path/to/folder --output_dir path/to/save --threshold 70.0
```
Or:
```bash
python main.py --input path/to/image.jpg --threshold 65.0
```

## 🧠 Parameters
| Argument       | Description                                             |
| -------------- | ------------------------------------------------------- |
| `--input_dir`  | Path to a directory of images                           |
| `--input`      | Path to a single image                                  |
| `--output_dir` | Where to save enhanced images (optional, auto fallback) |
| `--threshold`  | Luminance threshold (default = 70.0)                    |
| `positional`   | Alternative to `--input_dir` or `--input`               |

⚠️ Use only one input source at a time: `--input`, `--input_dir`, or `positional`.

## 🧪 Example Usage
```bash
# Detect and enhance images in a folder
python main.py --input_dir images/test_dataset

# Process a single image with custom threshold
python main.py --input images/sample.jpg --threshold 65.0

# Quick run using just positional argument
python main.py images/sample.jpg
```

## Project Structure
```
image-enhancer/
├── detector/
│   └── luminance.py              # Luminance-based image detection
├── logs/                         # Log files
├── main.py                       # Main runner script
├── utils.py                      # Logger and helper functions
├── requirements.txt              # Dependencies
└── README.md                     # You're reading it!
```

## 🧩 To-Do
- [x] Add low-light detection
- [x] Integrate Zero-DCE for enhancement
- [x] Integrate DeblurGAN (blurred image enhancement)
- [x] Auto-evaluation and quality metrics

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Credits
- [Zero-DCE by Chongyi Li](https://github.com/Li-Chongyi/Zero-DCE)
- [DeblurGAN by Orest Kupyn](https://github.com/KupynOrest/DeblurGAN)
- Built with ❤️ by ubeann
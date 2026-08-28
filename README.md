# Pre-Trained Image Classifier to Identify Dog Breeds

An end-to-end deep learning project comparing pre-trained Convolutional Neural Network (CNN) architectures (**VGG-16**, **ResNet-18**, and **AlexNet**) trained on ImageNet to accurately classify images of dogs, distinguish dogs from non-dogs, and identify dog breeds.

---

## 📌 Project Overview & Objectives

1. **Dog Identification**: Correctly identify whether an image depicts a dog or a non-dog entity (even if breed is misclassified).
2. **Breed Classification**: Accurately predict the exact breed of dog for all verified dog images.
3. **Architecture Comparison**: Benchmark three CNN architectures (**VGG**, **ResNet**, **AlexNet**) on accuracy and error patterns.
4. **Computational Tradeoff Analysis**: Evaluate runtime and memory overhead vs. classification accuracy to determine the optimal model for production constraints.

---

## 🗂️ Project Directory Structure

```text
image_classifier/
│
├── .venv/                              # Python virtual environment (isolated dependencies)
├── pet_images/                         # 40 benchmark images (30 dog breeds, 10 non-dogs)
├── uploaded_images/                    # Custom test images (e.g. flipped images, objects)
├── results/                            # Output folder storing all benchmark reports
│   ├── resnet_pet-images.txt           # Evaluation output for ResNet on pet_images/
│   ├── alexnet_pet-images.txt          # Evaluation output for AlexNet on pet_images/
│   ├── vgg_pet-images.txt              # Evaluation output for VGG on pet_images/
│   ├── resnet_uploaded-images.txt      # Evaluation output for ResNet on uploaded_images/
│   ├── alexnet_uploaded-images.txt     # Evaluation output for AlexNet on uploaded_images/
│   ├── vgg_uploaded-images.txt         # Evaluation output for VGG on uploaded_images/
│   └── check_images.txt                # Technical answers on uploaded images evaluation
│
├── check_images.py                     # Main orchestrator pipeline with timer & validation
├── classifier.py                       # Pretrained PyTorch inference engine (lazy loading)
├── get_input_args.py                   # Command-line argument parser (--dir, --arch, --dogfile)
├── get_pet_labels.py                   # Extracts ground-truth labels from image filenames
├── classify_images.py                  # Runs CNN inference and compares predictions with truth
├── adjust_results4_isadog.py           # Flags images as 'dog' or 'not-dog' using dognames.txt
├── calculates_results_stats.py         # Computes counts and percentage accuracy metrics
├── print_results.py                    # Formats summary tables and misclassification logs
├── test_classifier.py                  # Standalone test script for model inference
├── print_functions_for_lab_checks.py   # Lab assertion checks for pipeline validation
├── dognames.txt                        # Reference database of 225 valid dog breeds
├── imagenet1000_clsid_to_human.txt     # ImageNet class ID to label dictionary
├── requirements.txt                    # Project dependencies
├── run_models_batch.bat                # Windows batch runner for pet_images
├── run_models_batch.sh                 # Unix/macOS bash runner for pet_images
├── run_models_batch_uploaded.bat       # Windows batch runner for uploaded_images
├── run_models_batch_uploaded.sh        # Unix/macOS bash runner for uploaded_images
└── README.md                           # Project documentation
```

---

## 🔍 Module Breakdown & Architecture

| File | Purpose & Responsibilities |
| :--- | :--- |
| **`check_images.py`** | Main entrypoint. Measures program execution runtime, orchestrates the complete pipeline, and outputs final statistics. |
| **`classifier.py`** | Wraps `torchvision.models` (VGG16, ResNet18, AlexNet) using a lazy-loading proxy dictionary. Standardizes input tensors with ImageNet mean/std normalization and performs evaluation mode inference. |
| **`get_input_args.py`** | Implements `argparse` to parse `--dir` (default: `pet_images/`), `--arch` (default: `vgg`), and `--dogfile` (default: `dognames.txt`). |
| **`get_pet_labels.py`** | Parses image filenames (e.g. `Boston_terrier_02259.jpg` $\rightarrow$ `boston terrier`), strips numbers/extensions, and creates the ground truth dictionary. |
| **`classify_images.py`** | Invokes `classifier()`, formats model labels to lowercase/trimmed text, and records match flags (1/0) against ground truth. |
| **`adjust_results4_isadog.py`** | Checks ground truth and predicted labels against `dognames.txt` to determine if each is a dog (1/0). |
| **`calculates_results_stats.py`** | Calculates summary counts (`n_images`, `n_dogs_img`, `n_correct_breed`, etc.) and percentage metrics (`pct_correct_dogs`, `pct_correct_breed`, etc.). |
| **`print_results.py`** | Prints formatted summary tables and optional logs for misclassified dogs or breeds. |

---

## ⚙️ Environment Setup & Installation

### 1. Create a Virtual Environment

```bash
# Windows
py -3.11 -m venv .venv

# macOS / Linux
python3 -m venv .venv
```

### 2. Activate the Environment

```bash
# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Run Single Model Classification

```bash
# Run with default settings (VGG on pet_images/)
python check_images.py

# Run ResNet on pet_images/
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt

# Run AlexNet on pet_images/
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt

# Run VGG on uploaded_images/
python check_images.py --dir uploaded_images/ --arch vgg --dogfile dognames.txt
```

### Run Batch Evaluation (Saves to `results/`)

#### On Windows:
```cmd
# Benchmark all 3 models on pet_images/
run_models_batch.bat

# Benchmark all 3 models on uploaded_images/
run_models_batch_uploaded.bat
```

#### On Linux / macOS:
```bash
# Benchmark all 3 models on pet_images/
sh run_models_batch.sh

# Benchmark all 3 models on uploaded_images/
sh run_models_batch_uploaded.sh
```

---

## 📊 Benchmark Results (`pet_images/` 40 Images)

| Metric | VGG-16 | ResNet-18 | AlexNet |
| :--- | :---: | :---: | :---: |
| **Total Images** | 40 | 40 | 40 |
| **Dog Images / Non-Dog Images** | 30 / 10 | 30 / 10 | 30 / 10 |
| **% Correct "Dogs"** | **100.0%** | **100.0%** | **100.0%** |
| **% Correct "Not-a-Dog"** | **100.0%** | 90.0% | **100.0%** |
| **% Correct Dog Breed** | **93.3%** (28/30) | 90.0% (27/30) | 80.0% (24/30) |
| **% Overall Label Match** | **87.5%** | 82.5% | 75.0% |
| **Elapsed Runtime** | ~5 seconds | **~1 second** | **~1 second** |

---

## 💡 Key Findings & Model Recommendation

1. **Best Overall Architecture**: **VGG-16** is the superior model for this task. It achieved a flawless **100.0%** accuracy at distinguishing dogs from non-dogs, and the highest breed identification accuracy at **93.3%**.
2. **Best Speed / Efficiency Tradeoff**: **ResNet-18** ran ~5x faster than VGG while retaining high breed accuracy (**90.0%**), making it the optimal choice for resource-constrained or real-time mobile deployment.
3. **AlexNet Performance**: While AlexNet correctly identified 100% of dogs vs non-dogs, its breed classification accuracy was lower (**80.0%**).
4. **Flip Invariance**: When testing horizontally mirrored images (`Dog_01.jpg` vs `Dog_02.jpg`), all three architectures maintained consistent predictions.

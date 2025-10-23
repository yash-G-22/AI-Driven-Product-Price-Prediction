# 🧠 AI Driven Product Pricing – Multimodal Machine Learning Pipeline
## Amazon ML Challenge 2025

## 📝 Overview
Developed a **multimodal deep learning pipeline** for predicting optimal product prices using both **textual catalog content** and **product images**.  
The model integrates **Natural Language Processing (NLP)**, **Computer Vision (CV)**, and **tabular machine learning (LightGBM)** techniques to deliver accurate and explainable price estimations for large-scale e-commerce datasets.

---

## 🔍 Key Features & Methodology

### 🧩 Data Processing
- Processed and cleaned textual product descriptions using regex-based normalization.  
- Handled **75,000 product samples** with structured attributes, text, and image links.

### 💬 Text Feature Extraction
- Extracted **384-dimensional semantic embeddings** using `SentenceTransformer (all-MiniLM-L6-v2)`.  
- Compressed text embeddings using a **PyTorch Autoencoder**, reducing dimensions from `384 → 128` for computational efficiency.

### 🖼️ Image Feature Extraction
- Utilized pretrained **EfficientNet-B5 CNN** from `torchvision.models` for extracting deep image features.  
- Captured visual product attributes such as color, pattern, and texture.

### 🔗 Multimodal Fusion
- Combined text and image embeddings into a **unified feature representation**.  
- Normalized features using **StandardScaler** to prepare for model training.

### 🤖 Model Training
- Employed **LightGBM Regressor** for final regression-based price prediction.  
- Conducted **Optuna-based Bayesian hyperparameter tuning** (50 trials) to minimize validation MAE.  
- Implemented **early stopping** and **checkpointing** to avoid overfitting.

### 📈 Evaluation & Performance
- **MAE (Mean Absolute Error):** approx 14 on training data (75,000 products).  
- **SMAPE (Symmetric Mean Absolute Percentage Error):** approx 56% on test data (25,000 products).  
- Achieved strong generalization on unseen product categories.

### 💾 Deployment & Artifacts
- Saved final trained **LightGBM model** and **StandardScaler** using `joblib`.  
- Generated **Kaggle-compatible submission CSV** for predicted prices.

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **ML Frameworks** | PyTorch, LightGBM |
| **Optimization** | Optuna |
| **Data Processing** | pandas, numpy, tqdm |
| **NLP** | SentenceTransformer (MiniLM-L6-v2) |
| **CV** | EfficientNet-B5 (`torchvision.models`) |
| **Compression** | Autoencoder (PyTorch) |
| **Metrics** | MAE, SMAPE |

---

## 🚀 Highlights
- Designed a **complete multimodal ML pipeline** integrating text, image, and numerical data streams.  
- Implemented **autoencoder-based dimensionality reduction** to enhance model performance and reduce overfitting.  
- Automated **hyperparameter tuning** using Optuna for optimal performance.  
- Achieved **robust predictive accuracy** suitable for real-world **AI-driven dynamic pricing** systems in e-commerce.

---

## 📂 Output Files
- `lgbm_final_model.txt` – Final trained LightGBM model.  
- `standard_scaler.bin` – StandardScaler for feature normalization.  
- `submission1.csv` – Final submission file.  

---

## 🧩 Future Improvements
- Integrate additional metadata features such as brand, category, and seller reputation.  
- Experiment with multimodal transformers (e.g., CLIP, ViLT) for richer text–image alignment.  
- Deploy as an API service for real-time pricing recommendations.

---

## 🏆 Results Summary
| Metric | Dataset | Value |
|---------|----------|-------|
| **MAE** | Train (75K) | **14** | 
| **SMAPE** | Test (25K) | **56%** |

---

**Author:** Yash  
📚 *AI Driven Product Pricing – Multimodal ML Pipeline*  
✨ *AI that Sees and Reads — Powering the Future of Smart Product Pricing*  

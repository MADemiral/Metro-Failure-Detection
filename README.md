
# Metro Failure Detection – Fine-Tuning and Training

This repository is part of a senior project focused on predicting failures in **metro train compressor systems**, specifically the **Air Production Unit (APU)**. It uses real sensor data and machine learning models to detect potential anomalies before they cause operational issues.

This subdirectory (`train_finetune/`) includes scripts and models for training, fine-tuning, and evaluating anomaly detection algorithms using both **analog** and **digital** signals.



## ⚙️ Models Used

### 🔧 Analog Signals

* **Model:** One-Class SVM
* **Purpose:** Detect anomalies in pressure and temperature-related signals
* **Tuning:** Performed using Grid Search over `kernel`, `nu`, and `gamma`
* **Best Performance (Fine-tuned):**

  * Precision: 74.34%
  * Recall: 37.88%
  * F1 Score: 50.14%

### 💡 Digital Signals

* **Model:** Sparse Autoencoder (SAE)
* **Purpose:** Reconstruct input signal and measure anomaly via reconstruction error
* **Training Data:** Normal behavior signals
* **Best Performance:**

  * Precision: 95.00%
  * Recall: 100.00%
  * F1 Score: 97.00%

---

## 🧪 Dataset

* **Name:** MetroPT-3 (used in paper and public research)
* **Rows:** \~87,000
* **Features:**

  * 7 Analog: e.g., TP2, TP3, H1
  * 8 Digital: e.g., COMP, DV\_electric, MPG
  * 1 Timestamp column
* **Labels:** Manually labeled using known anomaly windows based on technical documentation.

---

## 🛠 Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Train Analog Model (One-Class SVM)

```bash
python train_finetune/train_finetune_ocsvm.py
```

### 2. Train Digital Model (Sparse Autoencoder)

```bash
python train_finetune/train_sae_digital.py
```

### 3. Train Addiional Models(Optional)

You can look to:
- `train_finetune/unlabeled_trainning.ipynb`


> Trained models will be saved to the `models/` directory.

---


## 📊 Explainability (Optional)

We use **SHAP** and **LIME** in separate notebooks/scripts to:

* Understand global feature importance
* Explain individual anomaly predictions
* Compare analog and digital signal behavior

---

## 👨‍💻 Authors

* **Mehmet Alp Demiral**
* **Mehmet Bumin Karacan**

> This project was developed as part of the CMPE 442 course at TED University.

---

## 📄 License

This repository is shared for educational and research purposes only. Please cite or reference if used in your own work.



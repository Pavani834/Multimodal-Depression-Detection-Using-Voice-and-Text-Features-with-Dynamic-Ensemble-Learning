# Multimodal-Depression-Detection-Using-Voice-and-Text-Features-with-Dynamic-Ensemble-Learning


## 📖 Overview
This project presents a multimodal approach for depression detection by combining **acoustic (speech)** and **linguistic (text)** features extracted from clinical interview data. The system utilizes **dynamic ensemble learning techniques (KNORA-U and KNORA-E)** to improve classification performance.

---

## 🚀 Key Features
- Multimodal analysis (Audio + Text)
- Acoustic feature extraction using **openSMILE**
- Linguistic feature extraction using **TF-IDF**
- Feature selection using **Fisher Score**
- Classification using:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Dynamic Ensemble Learning (KNORA-U, KNORA-E)

---

## 📊 Results
| Model | Accuracy |
|------|--------|
| Logistic Regression | 62.5% |
| SVM | 75.0% |
| Random Forest | 87.5% |
| KNORA-U | **91.78%** |
| KNORA-E (Proposed) | **91.78%** |

---

## 📂 Dataset
This work uses the **DAIC-WOZ dataset**, which contains clinical interview recordings and transcripts.

🔗 https://dcapswoz.ict.usc.edu/

> Note: Dataset is not included due to licensing restrictions.

---
##⚙️ Installation & Execution


git clone https://github.com/Pavani834/Multimodal-Depression-Detection-Using-Voice-and-Text-Features-with-Dynamic-Ensemble-Learning.git

cd Multimodal-Depression-Detection-Using-Voice-and-Text-Features-with-Dynamic-Ensemble-Learning

pip install -r requirements.txt

python main.py
---
🧠 Methodology
	1.	Data collection using DAIC-WOZ dataset
  2.	Preprocessing of audio and text data
	3.	Acoustic feature extraction using openSMILE
	4.	Text feature extraction using TF-IDF
	5.	Feature selection using Fisher Score
	6.	Feature fusion of audio and text modalities
	7.	Classification using machine learning models
	8.	Performance enhancement using dynamic ensemble techniques (KNORA-U and KNORA-E)

---

 📈 Output
	•	Accuracy and F1-score are displayed in the console
	•	Model performance evaluated using cross-validation

  ---

👤 Author

Pavani

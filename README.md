# 🩺 AI-Powered PCOS Early Screening

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.20+-red.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.12-orange.svg)
Preliminary risk assessment for Polycystic Ovary Syndrome (PCOS).

## 🚀 Key Features

* **AI Hair Density Analysis:** Utilizes a **ResNet-50** Convolutional Neural Network to analyze scalp images for signs of androgenetic alopecia (thinning), a common PCOS marker.
* **Clinical Risk Engine:** Integrates user data including BMI, menstrual cycle regularity, family history, and hormonal acne patterns.
* **Dynamic Visualizations:** Generates a real-time **Risk Factor Breakdown** graph using Matplotlib to show which factors contribute most to the user's score.
* **Personalized Recommendations:** Provides tailored lifestyle and dietary advice based on the calculated risk level (Low, Moderate, High).
* **Professional UI:** A clean, Teal-themed interface designed with a focus on **Human-Computer Interaction (HCI)** principles.

## 🛠️ Technical Architecture

### 1. Computer Vision (Image Analysis)
The system uses the **ResNet-50** architecture. The model extracts features from uploaded scalp images, which are then processed through a **Heuristic Output Normalization** layer to map AI confidence to a 10-50 point medical scale.

### 2. The Scoring Logic (Rotterdam Criteria Based)
The final risk percentage is calculated as follows:
| Factor | Max Points |
| :--- | :--- |
| AI Hair Density Analysis | 50 pts |
| Menstrual Cycle Regularity | 10 pts |
| BMI (Body Mass Index) | 10 pts |
| Hormonal Acne | 10 pts |
| Hirsutism (Excess Hair) | 10 pts |
| Family History | 10 pts |

Install dependencies:

pip install -r requirements.txt

Run the Application:

streamlit run sample.py

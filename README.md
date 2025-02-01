# WHALE-SOUND-ANALYSIS
# 🐋 Whale Call Detection and Classification

### 🌊 **Project Overview**  
This project focuses on detecting and classifying whale calls in oceanic environments using sound analysis techniques. It aims to analyze the various sounds produced by whales, compare the performance of traditional and modern sound classification methods, and assess the impact of anthropophonic noise on whale communication.

---

## 🔍 **Problem Statement**  
Whale calls are vital for communication, navigation, and mating among marine mammals. However, human-made (anthropophonic) noise from ships, industrial activities, and sonar can interfere with these calls, potentially causing harm to marine life. This project aims to build a sound detection and classification system to better understand whale communication and evaluate the effect of noise pollution on these mammals.

---

## 📂 **Project Structure**  
```plaintext
├── data/                   # Whale sound datasets
├── notebooks/              # Jupyter notebooks for analysis
├── models/                 # Pre-trained and custom models
├── results/                # Visualizations and performance metrics
└── README.md               # Project documentation
```

---

## ⚙️ **Tech Stack**  
- **Language**: Python  
- **Libraries**:  
  - NumPy  
  - Pandas  
  - Librosa (for audio analysis)  
  - Matplotlib & Seaborn (for visualizations)  
  - Scikit-learn  
  - TensorFlow/PyTorch (for deep learning models)  

---

## 🔧 **How to Run the Project**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/whale-call-classification.git
   cd whale-call-classification
   ```
2. Install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing and training scripts:  
   ```bash
   python src/preprocess.py  
   python src/train_model.py
   ```

---

## 📊 **Key Features**  
- **Sound Preprocessing**: Converts raw audio into spectrograms for better analysis.  
- **Classification Models**: Utilizes both traditional machine learning models and deep learning (CNNs) for sound classification.  
- **Noise Analysis**: Compares the impact of different noise levels on whale calls.  

---

## 📈 **Results**  
- Improved accuracy using CNNs for whale call classification.  
- Significant insights into how anthropophonic noise affects whale calls.  
- Performance metrics such as accuracy, precision, recall, and F1-score are documented.

---

## 🧪 **Future Scope**  
- Enhancing the dataset with more diverse whale sounds.  
- Implementing real-time sound detection using IoT devices.  
- Exploring noise cancellation techniques to mitigate the impact of human-made noise on whale communication.

---

## 📜 **License**  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 🤝 **Contributing**  
Contributions are welcome! Please open an issue or submit a pull request if you want to improve the project.

---

## 📧 **Contact**  
For any queries or collaboration opportunities, reach out to **Pranav Tiwari** at:  
- Email: tiwari.pranav1999@gmail.com
- LinkedIn: [linkedin.com/in/pranav-tiwari](https://www.linkedin.com/in/pranav-tiwari)  
- GitHub: [github.com/pranav-tiwari](https://github.com/pranav-tiwari)
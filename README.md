# 🧠 SimpleRNN Sentiment Analysis & Hidden State Visualizer

A complete end-to-end Machine Learning web application that performs **Multi-Class Sentiment Analysis** (`Positive`, `Negative`, `Neutral`) using a **Simple Recurrent Neural Network (SimpleRNN)**, served via a **FastAPI** backend REST API and an interactive **Streamlit** visualization frontend.

---

## 🌟 Features

* **Multi-Class Sentiment Classification**: Trained on social media tweet data (`positive`, `negative`, `neutral`).
* **Sequence Processing & Word Embeddings**: Keras Tokenization, padding, and continuous word embeddings.
* **RNN Hidden State Visualizer**: Real-time **Plotly Heatmap** visualizing intermediate hidden neuron activations at every timestep/word using weight-sharing interpretability (`return_sequences=True`).
* **FastAPI Backend Service**: High-performance REST endpoints (`/health`, `/predict`, `/dataset/stats`).
* **Streamlit Frontend**: Clean, responsive user interface with sample presets, class probability distributions, and sequence breakdown tables.

---

## 📁 Repository Structure

```text
├── backend/
│   └── main.py              # FastAPI REST API server
├── frontend/
│   └── app.py               # Streamlit interactive visualization UI
├── RNN_Implementation.ipynb # Colab-ready notebook for model training & export
├── train_model.py           # Standalone Python script for local training
├── sentiment_analysis.csv   # Dataset containing tweet texts and sentiment labels
├── requirements.txt         # Project dependencies
├── sentiment_rnn.weights.h5 # Trained model weights
├── tokenizer.pkl            # Pickled Keras Tokenizer
└── config.pkl               # Model configuration & label mappings
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model (Optional)
If you wish to re-train the model locally:
```bash
python train_model.py
```
*(Or run `RNN_Implementation.ipynb` in Google Colab / Jupyter).*

### 3. Launch FastAPI Backend
```bash
uvicorn backend.main:app --reload --port 8000
```
- API Documentation available at: `http://localhost:8000/docs`

### 4. Launch Streamlit Frontend
```bash
streamlit run frontend/app.py
```
- Open your browser at: `http://localhost:8501`

---

## 🧠 Model Architecture

| Layer | Output Shape | Parameters | Description |
| :--- | :--- | :---: | :--- |
| **Input** | `(None, 35)` | `0` | Padded sequence of word integer tokens |
| **Embedding** | `(None, 35, 32)` | `96,000` | Dense vector embedding (`mask_zero=True`) |
| **SimpleRNN** | `(None, 16)` | `784` | Recurrent layer with 16 hidden neurons |
| **Dense (Softmax)** | `(None, 3)` | `51` | Multi-class sentiment output probabilities |

---

## 🔬 How Hidden State Visualization Works

1. The primary SimpleRNN model processes the sequence and outputs the final hidden state $h_T$ for prediction.
2. A sequence-inspection model is constructed with `return_sequences=True`.
3. The trained weight matrices ($W_{xh}, W_{hh}, b_h$) are copied into the inspection model.
4. The inspection model outputs the activation matrix of shape `(Sequence Length, 16 Hidden Neurons)`, allowing step-by-step visualization of how each word influences internal RNN states.

---

## 📜 License
This project is open-source under the MIT License.

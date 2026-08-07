import os
import pickle
import traceback
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(
    title="RNN Sentiment Analysis API",
    description="FastAPI Backend for SimpleRNN Sentiment Analysis & Hidden State Visualization",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global variables for model artifacts
model = None
tokenizer = None
config = None
inspect_model = None

LABEL_MAPPING = {'negative': 0, 'neutral': 1, 'positive': 2}
INV_LABEL_MAPPING = {0: 'negative', 1: 'neutral', 2: 'positive'}
MAXLEN = 35

def build_model_architecture(vocab_size=3000, embed_dim=32, rnn_units=16, maxlen=35, num_classes=3):
    inp = Input(shape=(maxlen,), dtype="int32", name='input')
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True, name='embed')(inp)
    rnn = SimpleRNN(units=rnn_units, return_sequences=False, return_state=False, name='simple_rnn')(x)
    out = Dense(num_classes, activation='softmax', name='out')(rnn)
    
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def load_artifacts():
    global model, tokenizer, config, inspect_model, MAXLEN, INV_LABEL_MAPPING
    
    weights_path = os.path.join(BASE_DIR, "sentiment_rnn.weights.h5")
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    config_path = os.path.join(BASE_DIR, "config.pkl")
    
    if os.path.exists(tokenizer_path) and os.path.exists(weights_path):
        try:
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            
            vocab_size = 3000
            embed_dim = 32
            rnn_units = 16
            
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    config = pickle.load(f)
                MAXLEN = config.get("maxlen", MAXLEN)
                vocab_size = config.get("vocab_size", vocab_size)
                embed_dim = config.get("embed_dim", embed_dim)
                rnn_units = config.get("rnn_units", rnn_units)
                inv_map = config.get("inv_label_mapping", {})
                if inv_map:
                    INV_LABEL_MAPPING = inv_map
            
            # Re-build model architecture & load weights
            model = build_model_architecture(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                rnn_units=rnn_units,
                maxlen=MAXLEN,
                num_classes=len(INV_LABEL_MAPPING)
            )
            model.load_weights(weights_path)
            print("Successfully loaded model weights from sentiment_rnn.weights.h5")

            # Build sequence inspection model with return_sequences=True
            rnn_layer = model.get_layer("simple_rnn")
            seq_inp = Input(shape=(MAXLEN,), dtype="int32", name="seq_inp")
            seq_emb = model.get_layer("embed")(seq_inp)
            rnn_seq = SimpleRNN(units=rnn_units, return_sequences=True, name="rnn_seq")
            seq_hidden = rnn_seq(seq_emb)
            
            rnn_seq.set_weights(rnn_layer.get_weights())
            inspect_model = Model(inputs=seq_inp, outputs=seq_hidden)
            print("Successfully initialized hidden state inspection model!")
            
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            traceback.print_exc()

@app.on_event("startup")
def startup_event():
    load_artifacts()

class SentimentRequest(BaseModel):
    text: str

class TimestepState(BaseModel):
    timestep: int
    word: str
    token_id: int
    hidden_state: List[float]

class SentimentResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    tokens: List[str]
    token_ids: List[int]
    hidden_states: List[List[float]]
    timestep_details: List[TimestepState]
    is_model_loaded: bool

@app.get("/")
def root():
    return {
        "message": "RNN Sentiment Analysis API is running",
        "model_loaded": model is not None,
        "base_dir": BASE_DIR,
        "endpoints": ["/health", "/predict", "/dataset/stats"]
    }

@app.get("/health")
def health():
    if model is None or tokenizer is None:
        load_artifacts()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    global model, tokenizer, inspect_model, MAXLEN, INV_LABEL_MAPPING
    
    if model is None or tokenizer is None:
        load_artifacts()
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model artifacts not loaded. Checked BASE_DIR: {BASE_DIR}. Please ensure tokenizer.pkl and sentiment_rnn.weights.h5 exist."
        )
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    # Tokenize and pad sequence
    sequence = tokenizer.texts_to_sequences([text])[0]
    padded_seq = pad_sequences([sequence], maxlen=MAXLEN, padding="post", truncating="post")
    
    # Get model prediction
    probs = model.predict(padded_seq, verbose=0)[0]
    pred_class_idx = int(np.argmax(probs))
    pred_label = INV_LABEL_MAPPING.get(pred_class_idx, str(pred_class_idx))
    confidence = float(probs[pred_class_idx])
    
    prob_dict = {INV_LABEL_MAPPING.get(i, f"class_{i}"): float(prob) for i, prob in enumerate(probs)}
    
    # Get hidden state activations per timestep
    hidden_seq = inspect_model.predict(padded_seq, verbose=0)[0]
    
    # Reconstruct words for each sequence position
    inv_word_index = {v: k for k, v in tokenizer.word_index.items()}
    inv_word_index[0] = "<PAD>"
    
    tokens = []
    token_ids = [int(tid) for tid in padded_seq[0]]
    timestep_details = []
    
    for t_idx, tid in enumerate(token_ids):
        word = inv_word_index.get(tid, "<OOV>") if tid != 0 else "<PAD>"
        tokens.append(word)
        h_state = [round(float(val), 4) for val in hidden_seq[t_idx]]
        
        timestep_details.append(
            TimestepState(
                timestep=t_idx + 1,
                word=word,
                token_id=tid,
                hidden_state=h_state
            )
        )
    
    return SentimentResponse(
        text=text,
        prediction=pred_label,
        confidence=confidence,
        probabilities=prob_dict,
        tokens=tokens,
        token_ids=token_ids,
        hidden_states=[[round(float(val), 4) for val in row] for row in hidden_seq],
        timestep_details=timestep_details,
        is_model_loaded=True
    )

@app.get("/dataset/stats")
def dataset_stats():
    csv_path = os.path.join(BASE_DIR, "sentiment_analysis.csv")
    if not os.path.exists(csv_path):
        return {"error": "sentiment_analysis.csv not found"}
    
    try:
        df = pd.read_csv(csv_path)
        df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        platform_counts = df['Platform'].value_counts().to_dict() if 'Platform' in df.columns else {}
        time_counts = df['Time of Tweet'].value_counts().to_dict() if 'Time of Tweet' in df.columns else {}
        
        return {
            "total_rows": len(df),
            "columns": list(df.columns),
            "sentiment_counts": sentiment_counts,
            "platform_counts": platform_counts,
            "time_counts": time_counts,
            "sample_records": df.head(10).to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}

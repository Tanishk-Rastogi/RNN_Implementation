import os
import pickle
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout

def train_and_export():
    print("Loading dataset...")
    csv_path = "sentiment_analysis.csv"
    if not os.path.exists(csv_path):
        url = "https://raw.githubusercontent.com/Tanishk-Rastogi/Sentiment-Analysis-using-RNN/main/sentiment_analysis.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(csv_path)
    
    # Preprocess
    df = df.dropna(subset=['text', 'sentiment']).copy()
    df['text'] = df['text'].astype(str).str.strip()
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
    
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['sentiment'].map(label_mapping)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df['sentiment'].value_counts())
    
    # Tokenization & Padding
    vocab_size = 5000
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>', lower=True)
    tokenizer.fit_on_texts(df['text'])
    
    sequences = tokenizer.texts_to_sequences(df['text'])
    maxlen = 35
    print(f"Vocabulary Size: {len(tokenizer.word_index) + 1}")
    print(f"Max Sequence Length: {maxlen}")
    
    X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    y = df['label'].values
    
    # LSTM Architecture with Dropout for strong generalization
    embed_dim = 64
    lstm_units = 32
    num_classes = 3
    
    inp = Input(shape=(maxlen,), dtype="int32", name='input')
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True, name='embed')(inp)
    x = Dropout(0.2)(x)
    lstm = LSTM(units=lstm_units, return_sequences=False, return_state=False, name='lstm_layer')(x)
    x_out = Dropout(0.2)(lstm)
    out = Dense(num_classes, activation='softmax', name='out')(x_out)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training LSTM model...")
    model.fit(X, y, epochs=25, batch_size=16, validation_split=0.15, verbose=1)
    
    # Save artifacts
    model.save_weights("sentiment_lstm.weights.h5")
    print("Saved sentiment_lstm.weights.h5")
    
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Saved tokenizer.pkl")
    
    config = {
        'label_mapping': label_mapping,
        'inv_label_mapping': {v: k for k, v in label_mapping.items()},
        'maxlen': maxlen,
        'vocab_size': vocab_size,
        'lstm_units': lstm_units,
        'embed_dim': embed_dim
    }
    
    with open("config.pkl", "wb") as f:
        pickle.dump(config, f)
    print("Saved config.pkl")

if __name__ == "__main__":
    train_and_export()

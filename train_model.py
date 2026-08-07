import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense

def train_and_export():
    print("Loading dataset...")
    df = pd.read_csv("sentiment_analysis.csv")
    
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
    vocab_size = 3000
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text'])
    
    sequences = tokenizer.texts_to_sequences(df['text'])
    maxlen = max(len(seq) for seq in sequences)
    print(f"Max sequence length: {maxlen}")
    
    X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    y = df['label'].values
    
    # Model Architecture
    embed_dim = 32
    rnn_units = 16
    num_classes = 3
    
    inp = Input(shape=(maxlen,), dtype="int32", name='input')
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True, name='embed')(inp)
    rnn = SimpleRNN(units=rnn_units, return_sequences=False, return_state=False, name='simple_rnn')(x)
    out = Dense(num_classes, activation='softmax', name='out')(rnn)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training SimpleRNN model...")
    model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2, verbose=1)
    
    # Save artifacts
    model.save("sentiment_rnn.keras")
    model.save_weights("sentiment_rnn.weights.h5")
    print("Saved sentiment_rnn.keras & sentiment_rnn.weights.h5")
    
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Saved tokenizer.pkl")
    
    config = {
        'label_mapping': label_mapping,
        'inv_label_mapping': {v: k for k, v in label_mapping.items()},
        'maxlen': maxlen,
        'vocab_size': vocab_size,
        'rnn_units': rnn_units,
        'embed_dim': embed_dim
    }
    
    with open("config.pkl", "wb") as f:
        pickle.dump(config, f)
    print("Saved config.pkl")

if __name__ == "__main__":
    train_and_export()

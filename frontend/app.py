import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Configure Streamlit Page
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# FastAPI Backend URL
BACKEND_URL = "http://localhost:8000"

# Custom CSS for styling and hiding sidebar toggle
st.markdown("""
<style>
    /* Hide Streamlit Sidebar elements */
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #9333EA, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.25rem;
        text-align: center;
        border: 1px solid #84E1BC;
    }
    .sentiment-negative {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.25rem;
        text-align: center;
        border: 1px solid #F8B4B4;
    }
    .sentiment-neutral {
        background-color: #FEF08A;
        color: #713F12;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.25rem;
        text-align: center;
        border: 1px solid #FDE047;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🧠 Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time text sequence classification & probability distribution analysis</div>', unsafe_allow_html=True)

# Preset sample sentences
st.markdown("##### Quick Test Samples:")
col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
preset = ""
if col_preset1.button("Positive Sample"):
    preset = "What an amazing product, I absolutely love it!"
if col_preset2.button("Negative Sample"):
    preset = "Terrible experience, very bad quality and waste of money."
if col_preset3.button("Neutral Sample"):
    preset = "Just got back from work, eating lunch."
if col_preset4.button("Complex Sample"):
    preset = "I thought it would be bad, but it turned out surprisingly decent."

user_input = st.text_area(
    "Enter text to analyze sentiment:",
    value=preset if preset else "Today was a fantastic day, everything went perfectly!",
    height=100
)

analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

if analyze_btn and user_input:
    with st.spinner("Processing sequence..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json={"text": user_input},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                st.divider()
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 🎯 Prediction Result")
                    pred_label = data["prediction"].upper()
                    confidence = data["confidence"] * 100
                    
                    if pred_label == "POSITIVE":
                        st.markdown(f'<div class="sentiment-positive">😊 POSITIVE ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                    elif pred_label == "NEGATIVE":
                        st.markdown(f'<div class="sentiment-negative">😡 NEGATIVE ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="sentiment-neutral">😐 NEUTRAL ({confidence:.1f}%)</div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Probabilities Bar Chart
                    probs = data["probabilities"]
                    prob_df = pd.DataFrame({
                        "Sentiment": [k.capitalize() for k in probs.keys()],
                        "Probability": [v * 100 for v in probs.values()]
                    })
                    
                    fig_probs = px.bar(
                        prob_df,
                        x="Sentiment",
                        y="Probability",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"},
                        text_auto=".1f",
                        title="Class Probability Distribution (%)"
                    )
                    fig_probs.update_layout(showlegend=False, yaxis_range=[0, 100], height=300)
                    st.plotly_chart(fig_probs, use_container_width=True)
                
                with col2:
                    st.markdown("### 🔤 Sequence Tokenization")
                    tokens = [t for t in data["tokens"] if t != "<PAD>"]
                    st.write(f"**Non-Padded Word Count:** `{len(tokens)}`")
                    st.write(f"**Tokens:** `{ ' → '.join(tokens) }`")
                    
                    # Display Token IDs Table
                    token_df = pd.DataFrame(data["timestep_details"])
                    non_pad_df = token_df[token_df["word"] != "<PAD>"][["timestep", "word", "token_id"]]
                    st.dataframe(non_pad_df, use_container_width=True, hide_index=True)
            
            else:
                st.error(f"Backend returned error {response.status_code}: {response.json().get('detail')}")
        
        except Exception as e:
            st.error(f"Failed to connect to FastAPI backend: {e}")

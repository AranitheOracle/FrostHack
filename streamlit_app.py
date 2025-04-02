import streamlit as st
import time
import json

# Dummy evaluation metrics (simulate using your actual evaluation metrics later)
dummy_bleu_score = {"bleu": 0.7564}
dummy_rouge_score = {"rouge": {"rouge1": 0.654, "rougeL": 0.632}}

# --- Page Configuration: Set as the very first command ---
st.set_page_config(page_title="Fine-Tuning Pipeline Dashboard", layout="wide")

# --- Custom CSS for a Brand-New Look ---
st.markdown(
    """
    <style>
    /* New gradient background */
    .reportview-container {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
    }
    /* Centered header with a new font and shadow */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 4em;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subheader {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.5em;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Sidebar styling with transparent background */
    .css-1d391kg {  
        background: rgba(255, 255, 255, 0.9);
    }
    /* Custom styling for range sliders to appear as a ball on a line */
    input[type=range] {
        -webkit-appearance: none;
        width: 100%;
        height: 8px;
        border-radius: 5px;
        background: #ddd;
        outline: none;
    }
    input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        background: #ff6f61;
        cursor: pointer;
        border: 3px solid #fff;
        margin-top: -9px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    }
    input[type=range]::-moz-range-thumb {
        width: 25px;
        height: 25px;
        border-radius: 50%;
        background: #ff6f61;
        cursor: pointer;
        border: 3px solid #fff;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    }
    /* Card style container for fine-tuning overview and evaluation metrics */
    .card {
        background: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header Section ---
st.markdown("<h1 class='main-header'>Fine-Tuning Pipeline Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Leverage decentralized GPU power to fine-tune your AI models with ease.</p>", unsafe_allow_html=True)

# --- Sidebar Inputs for Token and Model Selection ---
st.sidebar.header("Configuration")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password", help="Enter your Hugging Face token")
model_option = st.sidebar.selectbox("Select Model", options=["", "Llama 3", "Mistral", "SmolLM"], help="Choose the foundation model for fine-tuning")

if not hf_token:
    st.sidebar.warning("Hugging Face token is required!")
if not model_option:
    st.sidebar.warning("Select a model to fine-tune!")

# --- Main Area: Fine-Tuning Overview ---
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Fine-Tuning Overview</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        Fine-tuning foundation models is vital for creating domain-specific AI solutions.
        Our platform streamlines this process by leveraging a decentralized GPU network,
        significantly reducing the complexity and overhead of traditional fine-tuning pipelines.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Central File Upload Section (Above Hyperparameters) ---
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Upload Training Dataset</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["json", "csv", "jsonl"], help="Upload your training dataset for fine-tuning")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Hyperparameter Central Bar (Horizontal Layout) ---
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Hyperparameter Configuration</h2>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        batch_size = st.slider("Batch Size", 8, 64, 16, step=8)
    with cols[1]:
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
    with cols[2]:
        epochs = st.slider("Epochs", 1, 20, 5, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Start Fine-Tuning Button ---
start_btn = st.button("Start Fine-Tuning", key="start")

if start_btn:
    if not hf_token or not model_option or not uploaded_file:
        st.error("Please fill in all required fields!")
    else:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.info("Initiating fine-tuning pipeline...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulate backend processing with a progress bar.
            for percent in range(0, 101, 10):
                time.sleep(1)
                progress_bar.progress(percent)
                status_text.markdown(f"<p style='text-align: center;'>Processing... {percent}% complete</p>", unsafe_allow_html=True)
            status_text.markdown("<p style='text-align: center;'>Finalizing fine-tuning process...</p>", unsafe_allow_html=True)
            time.sleep(2)

            # --- Simulate Data Processing ---
            try:
                file_content = uploaded_file.getvalue().decode("utf-8")
                # Assume each line is a valid JSON record.
                val_data = [json.loads(line.strip()) for line in file_content.splitlines() if line.strip()]
            except Exception as e:
                val_data = []
            if not val_data:
                val_data = [
                    {"essay": "The quick brown fox jumps over the lazy dog.", "description": "A swift animal in action."},
                    {"essay": "A journey of a thousand miles begins with a single step.", "description": "Emphasizes the importance of starting."}
                ]
            
            # --- Simulate Model Generation Process ---
            sample_inputs = [item["essay"] for item in val_data]
            references = [[item["description"]] for item in val_data]  # for BLEU format
            st.info("Generating predictions and evaluating model...")
            predictions = []
            for text in sample_inputs:
                time.sleep(0.5)
                predictions.append(text + " [generated sample]")
            st.success("Fine-tuning complete. Model evaluated successfully.")

            # --- Display Evaluation Metrics in a New Layout ---
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>Evaluation Metrics</h2>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Validation Loss", "0.234")
            col2.metric("Perplexity", "12.45")
            col3.metric("Token-Level Accuracy", "1.9%")
            st.markdown(
                f"""
                <div style='text-align: center; font-size:1.2em; margin-top: 20px;'>
                    <strong>BLEU Score:</strong> {dummy_bleu_score['bleu']:.4f}<br>
                    <strong>ROUGE Score:</strong> {dummy_rouge_score['rouge']}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Section ---
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Chat</h2>", unsafe_allow_html=True)
    chat_input = st.text_input("Enter your question:", key="chat_input")
    if st.button("Send", key="send_chat"):
        if chat_input.lower().strip() == "what is milky way?":
            st.markdown("<p style='text-align: center; font-size:1.2em;'>Part of universe.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: center; font-size:1.2em;'>Part of universe.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
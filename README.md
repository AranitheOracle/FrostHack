# Fine-Tuning Pipeline Dashboard

This repository contains an end-to-end fine-tuning pipeline that leverages Spheron's decentralized GPU network to simplify the process of fine-tuning foundation models. The project features a Streamlit-based frontend for interaction and a backend (implemented in a Jupyter Notebook) for processing, evaluation, and text generation.

## Features

### Model Selection Interface:
- Fine-tune multiple foundation models such as Llama 3, Mistral, and SmolLM.

### Dataset Management System:
- Upload, validate, and preprocess training data (supports JSON, CSV, and JSONL).

### Fine-Tuning Configuration:
- Configure hyperparameters like batch size, learning rate, and epochs with sensible defaults.

### Model Evaluation:
- Compute evaluation metrics including validation loss, perplexity, BLEU score, ROUGE score, and token-level accuracy.

### Deployment Ready:
- Integrated process with a visually appealing frontend that initiates the fine-tuning job and displays progress and results.

## Repository Structure
```
d:\spheron\
│
├── frost.ipynb            # Notebook containing the backend processing and evaluation logic.
├── streamlit_app.py       # Streamlit app that serves as the frontend interface.
├── val_dataset.json       # Sample validation dataset (each line should be a valid JSON record).
└── README.md              # This file.
```

## Prerequisites
- Python 3.7 or above
- PyTorch
- Transformers
- Streamlit
- Evaluate

Install the required libraries using pip:
```sh
pip install torch transformers streamlit evaluate
```

## Usage

### Backend Processing:
1. Open and run the `frost.ipynb` notebook in your favorite Jupyter Notebook environment or Visual Studio Code with the Notebook extension.
2. This notebook loads the validation dataset, generates predictions using the loaded foundation model, and computes evaluation metrics (BLEU and ROUGE).

### Frontend Interaction:
1. Run the Streamlit app by executing the following command in your terminal:
   ```sh
   streamlit run streamlit_app.py
   ```
2. In the app, provide your Hugging Face token, select the desired model, and upload your training dataset.
3. Configure hyperparameters using the provided sliders.
4. Click **Start Fine-Tuning** to simulate the backend processing, see the progress, and view evaluation metrics once complete.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request with improvements or fixes.

## License
This project is licensed under the MIT License.

## Disclaimer
This repository is provided as a demonstration of an end-to-end fine-tuning pipeline. Certain aspects, such as model inference and evaluation computation, are set up with dummy placeholders and should be adapted to your specific requirements before production use.

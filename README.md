# CPU Usage Prediction with MLflow & DVC

## 📌 Project Overview
This project predicts CPU usage from tabular system metrics using multiple machine learning models.  
It demonstrates:
- Data versioning with **DVC**
- Experiment tracking with **MLflow**
- Model comparison and evaluation
- Reproducible pipelines

---

## 📂 Repository Structure
cpu_usage_project/
│── data/ # Input datasets (DVC-tracked)
│── outputs_eval/ # Evaluation artifacts (plots, metrics)
│── src/ # Source code
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│── params.yaml # Hyperparameters & pipeline configs
│── dvc.yaml # DVC pipeline definition
│── requirements.txt # Python dependencies
│── README.md # Project documentation

## ⚙️ Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/keerthi152004/cpu_usage_project.git
   cd cpu_usage_project
2. Create a virtual environment and install dependencies:

    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows
    pip install -r requirements.txt


3. Pull datasets from DVC remote:
   dvc pull


## Running the Pipeline

    Run the full pipeline (preprocessing → training → evaluation):

    dvc repro


    Or run individual steps:

    python src/preprocess.py
    python src/train.py
    python src/evaluate.py


## Experiment Tracking

    All experiments are logged in MLflow.

    To view MLflow UI locally:

    mlflow ui


    Open in browser: http://localhost:5000



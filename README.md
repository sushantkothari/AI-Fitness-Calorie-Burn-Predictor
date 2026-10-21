# AI Fitness Calorie Burn Predictor

## Overview

The AI Fitness Calorie Burn Predictor is a production-focused machine learning system designed to estimate calorie expenditure during physical activity using physiological and exercise-related parameters. The project combines robust feature engineering, deterministic preprocessing pipelines, and optimized feature selection with a lightweight Flask-based inference service.

The architecture follows practical machine learning engineering and deployment principles, ensuring consistency between training and inference environments while maintaining modularity, scalability, and reproducibility. The system is suitable for real-world fitness analytics applications, applied research, and production-oriented ML experimentation.

---

## Key Highlights

- End-to-end supervised learning pipeline for calorie burn estimation
- Deterministic preprocessing pipeline serialized for deployment consistency
- Recursive Feature Elimination (RFE) for dimensionality reduction and improved generalization
- Modular separation between model training and inference layers
- Lightweight Flask application enabling real-time prediction
- Production-ready artifact management using serialized pipelines
- Designed to minimize training-serving skew and ensure reproducible inference

---

## System Architecture

### Training Pipeline

Model development and experimentation are implemented in `Advanced_calorie_prediction.ipynb`. The workflow follows a structured machine learning lifecycle:

1. Data ingestion and exploratory analysis
2. Data validation, cleaning, and preprocessing
3. Feature engineering and transformation
4. Numerical scaling and normalization
5. Recursive Feature Elimination for optimal feature subset selection
6. Model training, validation, and performance benchmarking
7. Serialization of preprocessing and predictive components

Generated artifacts:

- `preprocessor.pkl` — Encapsulates preprocessing logic and feature transformations
- `rfe_model.pkl` — Final trained regression model

Encapsulation of preprocessing ensures deterministic transformations during inference.

---

### Deployment Architecture

The deployment layer is implemented using Flask to provide a lightweight and efficient inference interface.

Execution flow:

1. Users provide structured fitness-related inputs through the web interface.
2. The Flask backend validates and structures incoming requests.
3. Input features are processed through the serialized preprocessing pipeline.
4. The trained model generates calorie burn predictions.
5. Predictions are rendered dynamically via server-side templates.

This architecture facilitates low-latency inference while ensuring strict consistency with the training pipeline.

---

## Machine Learning Methodology

### Data Preprocessing Strategy

The preprocessing pipeline standardizes transformations to ensure reliability across environments. Key responsibilities include:

- Numerical feature scaling and normalization
- Consistent feature ordering
- Transformation of engineered attributes
- Schema alignment between training and inference

Embedding preprocessing within a serialized pipeline eliminates inconsistencies and reduces operational risk.

### Feature Selection

Recursive Feature Elimination is applied to enhance model efficiency and robustness by:

- Removing redundant or low-importance features
- Reducing overfitting risk
- Improving model interpretability
- Optimizing computational performance

### Model Persistence and Inference

The preprocessing pipeline and trained model are serialized using Pickle, enabling:

- Rapid application startup
- Lightweight deployment without retraining
- Clear separation between experimentation and production environments

---

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sushantkothari/AI-Fitness-Calorie-Burn-Predictor.git
    cd AI-Fitness-Calorie-Burn-Predictor
    ```

2. Create and activate a virtual environment:

    ```bash
    # Windows (PowerShell / CMD)
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    # If a requirements.txt file is present (recommended)
    pip install -r requirements.txt

    # Minimal core dependencies (fallback)
    pip install flask scikit-learn pandas numpy jupyter
    ```

---

## Running the Application

1. Navigate to the Flask application directory:

    ```bash
    cd flask_app
    ```

2. Start the server:

    ```bash
    python app.py
    ```

3. Open the application in a browser:

    ```
    http://127.0.0.1:5000/
    ```

---

## Input Interface

The model accepts structured physiological and workout-related inputs including:

- Age
- Gender
- Height
- Weight
- Heart Rate
- Body Temperature
- Exercise Duration
- Engineered features generated during preprocessing

All inputs are automatically validated and transformed before inference.

---

## Technology Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Jinja2 Templates
- Pickle Serialization

---

## Engineering Principles

- Reproducible machine learning workflows
- Separation of concerns between training and serving layers
- Deterministic preprocessing to prevent training-serving skew
- Modular and extensible architecture
- Lightweight inference design optimized for deployment

---

## Potential Extensions

- RESTful API endpoints for external system integration
- Containerization using Docker for scalable deployment
- Model interpretability integration (SHAP or LIME)
- Cloud deployment pipelines
- Comparative benchmarking with deep learning architectures
- Automated monitoring and performance tracking

---

## License

This project is licensed under the [LICENSE](LICENSE). See the LICENSE file for details.

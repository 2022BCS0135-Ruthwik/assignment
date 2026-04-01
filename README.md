# MLOps Pipeline Assignment

## Student Information
- **Name**: <YOUR_NAME>
- **Roll No**: <YOUR_ROLLNO>

## Project Overview
End-to-end MLOps pipeline for the Wine Quality Dataset, featuring data versioning (DVC), experiment tracking (MLflow), model training (Linear Regression, Random Forest), API deployment (FastAPI), Docker containerization, and CI/CD (GitHub Actions).

## Prerequisites
- Python 3.9+
- Docker
- Git
- AWS Account (for S3 bucket, used with DVC)

## Step-by-Step Instructions

### 1. Local Setup
```bash
git clone https://github.com/<ROLLNO>-mlops-assignment
cd project
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. DVC Setup and Data Versioning
```bash
# Initialize DVC
dvc init

# Add S3 remote
dvc remote add -d myremote s3://my-mlops-bucket/<ROLLNO>

# Download and split dataset
python src/data_prep.py

# Add data to DVC
dvc add data/v1_dataset.csv
dvc add data/v2_dataset.csv

# Push to S3 remote (Requires AWS credentials to be configured)
dvc push
```

### 3. Training and MLflow Tracking
You can run the different experiments as requested:
```bash
# Run 1: Version 1 + Linear Regression (baseline)
python src/train.py --data_path data/v1_dataset.csv --model_type lr --run_desc "Run 1: V1 + LR (Baseline)"

# Run 2: Version 1 + Linear Regression (different hyperparameter)
python src/train.py --data_path data/v1_dataset.csv --model_type lr_hyper --run_desc "Run 2: V1 + LR (Hyper)"

# Run 3: Version 2 + Linear Regression
python src/train.py --data_path data/v2_dataset.csv --model_type lr --run_desc "Run 3: V2 + LR"

# Run 4: Version 2 + feature selection
python src/train.py --data_path data/v2_dataset.csv --model_type lr --subset_features --run_desc "Run 4: V2 + Feature Selection"

# Run 5: Version 2 + Random Forest
python src/train.py --data_path data/v2_dataset.csv --model_type rf --run_desc "Run 5: V2 + RF"
```

To run the MLflow UI and view experiments:
```bash
mlflow ui
# Open browser at http://127.0.0.1:5000
```

### 4. Run FastAPI Locally
```bash
uvicorn app.main:app --reload
```
Test the endpoints at `http://127.0.0.1:8000/` and `http://127.0.0.1:8000/predict`.

### 5. Docker Deployment
```bash
# Build Image
docker build -t <DOCKER_USERNAME>/<YOUR_ROLLNO>-mlops .

# Run Container
docker run -p 8000:8000 <DOCKER_USERNAME>/<YOUR_ROLLNO>-mlops

# Push Image
docker push <DOCKER_USERNAME>/<YOUR_ROLLNO>-mlops
```

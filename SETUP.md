# Setup Instructions

## Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Notebook

```bash
# Start Jupyter
jupyter notebook

# Or use VS Code with Python extension
code .
```

## For Google Colab

Upload `requirements.txt` and run in first cell:
```python
!pip install -r requirements.txt
```

Then upload `data_preprocessing/combined_banks_dataset.csv` and change path in code to:
```python
data = pd.read_csv("combined_banks_dataset.csv")
```

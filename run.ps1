# Setup script to create virtual environment, install requirements, and run the Streamlit app

$PythonPath = "python"

# Verify if python works, otherwise use explicitly installed winget path
if (-not (Get-Command python -ErrorAction SilentlyContinue) -or (&$PythonPath --version 2>&1) -match "Python was not found") {
    $PythonPath = "C:\Users\ELCOT\AppData\Local\Programs\Python\Python311\python.exe"
}

Write-Host "Using Python path: $PythonPath" -ForegroundColor Green

if (Test-Path $PythonPath) {
    Write-Host "Setting up virtual environment..."
    &$PythonPath -m venv venv
    
    Write-Host "Activating virtual environment..."
    .\venv\Scripts\activate
    
    # After activation, 'python' and 'pip' should refer to the venv
    Write-Host "Installing required tools (Streamlit, LangChain, etc.)..."
    .\venv\Scripts\python.exe -m pip install -r requirements.txt
    
    Write-Host "Installing Spacy / SciSpacy models (this might take a minute)..."
    .\venv\Scripts\python.exe -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
    
    Write-Host "Starting Factual Hallucination Detection Streamlit App..."
    .\venv\Scripts\streamlit.exe run app.py
} else {
    Write-Host "❌ Python is not installed or not in your PATH. Please install Python from https://www.python.org/downloads/ and try again." -ForegroundColor Red
}

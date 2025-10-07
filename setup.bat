@echo off
SETLOCAL

echo ---------------------------------------
echo Setting up HAS environment...
echo ---------------------------------------

REM --- Check Python ---
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.10+ first.
    exit /b 1
)

REM --- Create virtual environment ---
IF NOT EXIST ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
) ELSE (
    echo Virtual environment already exists.
)

REM --- Activate venv ---
call .venv\Scripts\activate.bat

REM --- Upgrade pip ---
echo Upgrading pip...
python -m pip install --upgrade pip

REM --- Install requirements ---
echo Installing Python packages...
python -m pip install --upgrade -r requirements.txt

REM --- Create models folder ---
IF NOT EXIST "models" mkdir models

REM --- Download Mistral model ---
echo Downloading Mistral model...
powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf?download=true' -OutFile 'models\mistral-7b-instruct-v0.2.Q4_K_M.gguf'"

echo ---------------------------------------
echo Setup complete! Run the server with:
echo .venv\Scripts\activate.bat && python app.py
echo ---------------------------------------
pause

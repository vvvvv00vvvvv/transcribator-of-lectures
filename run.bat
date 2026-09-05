@echo off
chcp 65001 > nul
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" goto ACTIVATE

echo Виртуальное окружение не найдено. Создание...
if exist ".venv" rmdir /s /q .venv

py -3.12 -m venv .venv
if not exist ".venv\Scripts\activate.bat" (
    echo Ошибка: Python 3.12 не найден. Убедитесь, что установлен Python 3.12.
    pause
    exit /b
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo NVIDIA GPU найден. Установка библиотек PyTorch и CUDA...
    pip install torch --index-url https://download.pytorch.org/whl/cu130
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
) else (
    echo GPU не найден. Установка CPU-версии PyTorch...
    pip install torch
)

pip install faster-whisper static-ffmpeg
goto RUN

:ACTIVATE
call .venv\Scripts\activate.bat

:RUN
echo Запуск транскрибатора...
python main.py
pause
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

echo Установка PyTorch c поддержкой CUDA 12.1...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper static-ffmpeg
goto RUN

:ACTIVATE
call .venv\Scripts\activate.bat

:RUN
echo Запуск транскрибатора...
python main.py
pause
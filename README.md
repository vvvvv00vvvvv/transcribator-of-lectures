# Транскрибатор Аудио/Видео
Автоматическая транскрибация медиафайлов с помощью `faster-whisper` и разбиением текста на части для передачи в LLM.
### Ввод
На вход программе подается медиафайл для транскрибации

### Вывод
На выходе в той же папке, что и медиафайл, получаем текстовые файлы:
- <название медифайла> raw transcript.txt
- <название медифайла> raw transcript_part_1.txt
- <название медифайла> raw transcript_part_2.txt

...

(по умолчанию исходный текст делится на 4 части)

---

## Требования
- Наличие Python 3.12 на ПК

## Установка и запуск
### 1. Клонирование репозитория
```bash
git clone https://github.com/vvvvv00vvvvv/transcribator-of-lectures.git
cd transcribator-of-lectures
```

### 2. Запуск
#### Windows
Запустите файл `run.bat` двойным кликом или через консоль:
```cmd
.\run.bat
```

#### Linux (с NVIDIA GPU / CUDA 12.1)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install faster-whisper static-ffmpeg
python main.py
```

#### macOS / Linux (без NVIDIA GPU / обработка на CPU)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch faster-whisper static-ffmpeg
python main.py
```

---

### 3. Улучшение текста
Для улучшения текста и избавления от ошибок транскрибации вставьте этот промпт в вашу нейросеть и добавьте текст из созданных файлов:

`<название исходного медиафайла> raw transcript_part_*`

[Промпт для LLM](./Prompt_for_LLM.md)

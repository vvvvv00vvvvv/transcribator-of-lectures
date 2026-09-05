from pathlib import Path
import datetime
import os
import sys
import torch

# 1. Автоматическое определение базового пути к Python (даже внутри .venv)
base_python_dir = sys.base_prefix

# 2. Автоматическое определение папок Tcl/Tk
tcl_dir = os.path.join(base_python_dir, "tcl", "tcl8.6")
tk_dir = os.path.join(base_python_dir, "tcl", "tk8.6")

if os.path.exists(tcl_dir):
    os.environ["TCL_LIBRARY"] = tcl_dir
if os.path.exists(tk_dir):
    os.environ["TK_LIBRARY"] = tk_dir

# 3. Подключение библиотек CUDA в Windows для CTranslate2 (cublas64_12.dll и cudnn)
if sys.platform == "win32":
    cuda_paths = [
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "lib"),
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            os.add_dll_directory(path)
            os.environ["PATH"] = path + os.path.pathsep + os.environ.get("PATH", "")

import static_ffmpeg
from faster_whisper import WhisperModel

static_ffmpeg.add_paths()

def split_file_with_overlap(file_path, num_parts=4, overlap=3):
    """
    Делит файл на num_parts частей с наложением строк.
    overlap: количество строк для захвата из предыдущей и следующей части.
    """
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return

    # 1. Читаем все строки файла в память (безопасно для файлов до нескольких сотен МБ)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)

    # Если файл пустой
    if total_lines == 0:
        print("Файл пуст.")
        return

    # 2. Вычисляем базовый размер части и остаток
    base_chunk_size = total_lines // num_parts
    remainder = total_lines % num_parts

    base_name, extension = os.path.splitext(file_path)

    current_start_idx = 0

    for i in range(num_parts):
        # 3. Определяем "ядро" текущей части (без наложения)
        # Распределяем остаток по первым частям, чтобы выровнять размеры
        current_part_size = base_chunk_size + (1 if i < remainder else 0)

        current_end_idx = current_start_idx + current_part_size

        # 4. Вычисляем границы с учетом наложения (overlap)
        # Для начала: берем overlap назад, но не меньше 0
        overlap_start = max(0, current_start_idx - overlap)

        # Для конца: берем overlap вперед, но не больше общего кол-ва строк
        overlap_end = min(total_lines, current_end_idx + overlap)

        # Извлекаем нужный срез строк
        chunk_lines = lines[overlap_start:overlap_end]

        # 5. Сохраняем в файл
        output_filename = f"{base_name}_part_{i + 1}{extension}"

        with open(output_filename, 'w', encoding='utf-8') as out_f:
            out_f.writelines(chunk_lines)

        print(f"Создан: {output_filename} | Строк: {len(chunk_lines)} "
              f"(Диапазон индексов: {overlap_start}-{overlap_end})")

        # Сдвигаем начало "ядра" для следующей итерации
        current_start_idx = current_end_idx

def select_file():
    # Попытка вызвать графическое окно Tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Выберите видео или аудиофайл",
            filetypes=[("Медиафайлы", "*.mp4 *.mkv *.avi *.mp3 *.wav *.m4a"), ("Все файлы", "*.*")]
        )
        if file_path:
            return file_path
    except Exception:
        pass

    # Фоллбэк: если Tkinter не работает, пользователь перетаскивает файл в окно консоли
    user_input = input("Перетащите видеофайл в окно консоли и нажмите Enter: ")
    return user_input.strip('"').strip("'")

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds))).zfill(8)

def main():
    file_path_str = select_file()
    if not file_path_str:
        print("Файл не выбран.")
        return

    video_path = Path(file_path_str)
    text_path = video_path.with_name(f"{video_path.stem}.raw transcript.txt")

    model_size = "large-v3"
    if torch.cuda.is_available():
        device = "cuda"
        free_bytes, _ = torch.cuda.mem_get_info()
        free_vram_gb = free_bytes / (1024 ** 3)

        if free_vram_gb >= 5.0:
            model_size = "large-v3"
            compute_type = "float16"
        elif free_vram_gb >= 3.0:
            model_size = "large-v3-turbo"
            compute_type = "float16"
            print("Мало VRAM для large-v3. Выбрана оптимизированная модель large-v3-turbo.")
        else:
            model_size = "medium"
            compute_type = "int8"
            print("Доступно менее 3 ГБ VRAM. Автоматический переподбор на модель 'medium' в режиме int8.")
    else:
        device = "cpu"
        compute_type = "int8"
        print("GPU не найден или CUDA недоступна. Запуск обработки на CPU.")

    print(f"Загрузка модели {model_size}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("Запуск транскрибации...")
    segments, info = model.transcribe(
        str(video_path),
        vad_filter=True,
        beam_size=5
    )

    print(f"Язык: '{info.language}' (вероятность {info.language_probability:.2f})")

    with open(text_path, "w", encoding="utf-8") as f:
        for segment in segments:
            start_t = format_time(segment.start)

            end_t = format_time(segment.end)
            # Формируем строку: [00:00:05 -> 00:00:10] Текст сообщения

            line = f"[{start_t} -> {end_t}] {segment.text}"
            # line = f"{segment.text}"
            print(line)
            f.write(line + "\n")

    print(f"\n\nГотово! Текст сохранен в: {text_path}")
    num_parts = 4
    overlap = 3
    print(f"\nЗапускаем разделение текста на части (для удобного экспорта в LLM для чистовой обработки)")
    print(f"\nРазделение на {num_parts} части c перекрытием каждого файла по {overlap} строки")
    split_file_with_overlap(str(text_path), num_parts=num_parts, overlap=overlap)

    print("FINISH")

if __name__ == "__main__":
    main()
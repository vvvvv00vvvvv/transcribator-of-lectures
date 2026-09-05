from pathlib import Path
import os
import sys

# 1. Автоматическое определение базового пути к Python (даже внутри .venv)
base_python_dir = sys.base_prefix

# 2. Автоматическое определение папок Tcl/Tk
tcl_dir = os.path.join(base_python_dir, "tcl", "tcl8.6")
tk_dir = os.path.join(base_python_dir, "tcl", "tk8.6")

if os.path.exists(tcl_dir):
    os.environ["TCL_LIBRARY"] = tcl_dir
if os.path.exists(tk_dir):
    os.environ["TK_LIBRARY"] = tk_dir

# 3. Подключение библиотек CUDA из PyTorch и NVIDIA в системный PATH
try:
    import torch
    torch_dir = os.path.dirname(torch.__file__)
    nvidia_dir = os.path.join(os.path.dirname(torch_dir), 'nvidia')

    torch_lib = os.path.join(torch_dir, 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + os.path.pathsep + os.environ['PATH']

    if os.path.exists(nvidia_dir):
        for root, dirs, files in os.walk(nvidia_dir):
            if 'lib' in dirs:
                lib_path = os.path.join(root, 'lib')
                os.add_dll_directory(lib_path)
                os.environ['PATH'] = lib_path + os.path.pathsep + os.environ['PATH']
except Exception:
    pass

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

def main():
    file_path_str = select_file()
    if not file_path_str:
        print("Файл не выбран.")
        return

    video_path = Path(file_path_str)
    text_path = video_path.with_name(f"{video_path.stem}.raw transcript.txt")

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print("Используется видеокарта NVIDIA (CUDA).")
    else:
        device = "cpu"
        compute_type = "int8"
        print("GPU не найден или CUDA не доступна. Переключение на CPU.")

    model_size = "large-v3"
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
            line = f"{segment.text}"
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
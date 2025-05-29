import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import tkinter.ttk as ttk
import threading
import os
import logging
from main_logic import process_images_and_generate_report

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Анализ графических символов")
        root.geometry("1000x700")

        # Основные фреймы
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10, fill=tk.X)

        # Параметры
        tk.Label(control_frame, text="Ширина:").grid(row=0, column=0)
        self.width_var = tk.Entry(control_frame, width=5)
        self.width_var.grid(row=0, column=1)
        self.width_var.insert(0, "5")

        tk.Label(control_frame, text="Высота:").grid(row=0, column=2)
        self.height_var = tk.Entry(control_frame, width=5)
        self.height_var.grid(row=0, column=3)
        self.height_var.insert(0, "7")

        # Кнопки
        ttk.Button(control_frame, text="Загрузить изображения",
                   command=self.add_images).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Анализировать",
                   command=self.start_analysis).grid(row=0, column=5)

        # Область вывода
        self.output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Статус
        self.status_var = tk.StringVar(value="Готов к работе")
        tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, side=tk.BOTTOM)

        self.image_paths = []

    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Изображения", "*.png;*.jpg;*.bmp")])
        if files:
            self.image_paths = list(files)
            self.output_text.insert(tk.END, f"Загружено изображений: {len(self.image_paths)}\n")
            self.status_var.set(f"Загружено изображений: {len(self.image_paths)}")

    def start_analysis(self):
        if len(self.image_paths) < 2:
            messagebox.showerror("Ошибка", "Требуется минимум 2 изображения")
            return

        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            size = (width, height)

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Начало анализа...\n")
            self.status_var.set("Анализ запущен")

            # Запуск в отдельном потоке
            threading.Thread(
                target=self.run_analysis,
                args=(size,),
                daemon=True
            ).start()

        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные размеры матрицы")

    def run_analysis(self, size):
        try:
            output_pdf = "analysis_report.pdf"

            def progress_callback(message):
                self.output_text.insert(tk.END, message)
                self.output_text.see(tk.END)
                self.root.update()

            results = process_images_and_generate_report(
                self.image_paths,
                output_pdf,
                size,
                progress_callback=progress_callback
            )

            self.output_text.insert(tk.END, "\nАнализ завершён!\n")
            self.status_var.set(f"Отчёт сохранён: {os.path.abspath(output_pdf)}")
            messagebox.showinfo("Готово", f"Отчёт сохранён:\n{os.path.abspath(output_pdf)}")

        except Exception as e:
            self.output_text.insert(tk.END, f"\nОшибка: {str(e)}\n")
            self.status_var.set("Ошибка анализа")
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
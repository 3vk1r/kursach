import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import tkinter.ttk as ttk

import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg
import threading
import os
import logging
from utils import load_image_as_matrix_and_vector, format_matrix, format_vector
from main_logic import process_images_and_generate_report

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class App:
    def __init__(self, root):
        self.root = root
        root.title("Анализ качества распознавания")
        root.geometry("1200x800")
        logger.info("Инициализация интерфейса")

        # Валидация
        self.validate_cmd = root.register(self.validate_size)

        # Панель управления
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10, fill=tk.X)

        # Поля размеров
        tk.Label(control_frame, text="Ширина (3-100):").grid(row=0, column=0)
        self.width_var = tk.Entry(control_frame, width=5, validate="key",
                                validatecommand=(self.validate_cmd, '%P'))
        self.width_var.grid(row=0, column=1)
        self.width_var.insert(0, "5")

        tk.Label(control_frame, text="Высота (3-100):").grid(row=0, column=2)
        self.height_var = tk.Entry(control_frame, width=5, validate="key",
                                 validatecommand=(self.validate_cmd, '%P'))
        self.height_var.grid(row=0, column=3)
        self.height_var.insert(0, "7")

        # Поле уровня шума
        tk.Label(control_frame, text="Уровень шума:").grid(row=0, column=4)
        self.noise_scale = tk.DoubleVar(value=3.0)
        ttk.Scale(control_frame, from_=0.1, to=5.0, variable=self.noise_scale,
                 orient=tk.HORIZONTAL, length=100).grid(row=0, column=5)
        ttk.Label(control_frame, textvariable=self.noise_scale).grid(row=0, column=6)

        # Кнопки
        ttk.Button(control_frame, text="Добавить изображения",
                  command=self.add_images).grid(row=0, column=7, padx=5)
        ttk.Button(control_frame, text="Анализировать",
                  command=self.start_analysis).grid(row=0, column=8)

        # Область миниатюр
        self.thumb_frame = tk.Frame(root)
        self.thumb_frame.pack(pady=10, fill=tk.X)
        self.thumbnails = []
        self.image_paths = []

        # Основная область
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка с матрицами
        self.matrix_tab = tk.Frame(self.notebook)
        self.matrix_text = scrolledtext.ScrolledText(self.matrix_tab, wrap=tk.WORD, font=('Courier New', 10))
        self.matrix_text.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.matrix_tab, text="Матрицы")

        # Вкладка с графиками
        self.graph_tab = tk.Frame(self.notebook)
        self.figure = plt.figure(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_tab)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.graph_tab, text="Графики")

        # Подсветка
        self.matrix_text.tag_config('highlight', foreground='red')
        self.report_path = tk.StringVar()
        tk.Label(root, text="Путь к отчету:").pack()
        tk.Entry(root, textvariable=self.report_path, width=50).pack()
        self.image_count_label = tk.Label(root, text="Загружено изображений: 0", fg="red")
        self.image_count_label.pack(pady=5)

        # Кнопка очистки
        self.clear_button = tk.Button(root, text="Очистить все", command=self.clear_all)
        self.clear_button.pack(pady=5)

    def clear_all(self):
        """Очистка всех загруженных изображений"""
        for lbl in self.thumbnails:
            lbl.destroy()
        self.thumbnails = []
        self.image_paths = []
        self.matrix_text.delete(1.0, tk.END)
        self.update_image_count()

    def validate_size(self, value):
        """Валидация ввода размеров"""
        if not value:
            return True
        try:
            num = int(value)
            return 1 <= num <= 100
        except ValueError:
            return False

    def add_images(self):
        """Добавление изображений с защищенным форматированием"""
        files = filedialog.askopenfilenames(filetypes=[("Изображения", "*.png;*.jpg;*.bmp")])
        if not files:  # Если пользователь отменил выбор
            return

        for path in files:
            if path not in self.image_paths:
                try:
                    # Загрузка изображения
                    size = (int(self.width_var.get()), int(self.height_var.get()))
                    matrix, vector = load_image_as_matrix_and_vector(path, size)

                    # Безопасное форматирование
                    matrix_str = format_matrix(matrix)
                    vector_str = format_vector(vector)

                    # Вывод в текстовое поле
                    self.matrix_text.insert(tk.END, f"Изображение {len(self.image_paths) + 1}:\n")
                    self.matrix_text.insert(tk.END, f"Матрица:\n{matrix_str}\n")
                    self.matrix_text.insert(tk.END, "Вектор:\n")

                    # Подсветка
                    for val in vector_str.split('\n'):
                        if val.strip() == '1':
                            self.matrix_text.insert(tk.END, val + '\n', 'highlight')
                        else:
                            self.matrix_text.insert(tk.END, val + '\n')

                    self.image_paths.append(path)
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка обработки {path}:\n{str(e)}")

        self.update_image_count()
        self.matrix_text.see(tk.END)  # Прокручиваем до конца

    def update_image_count(self):
        """Обновление счетчика изображений"""
        count = len(self.image_paths)
        self.image_count_label.config(text=f"Загружено изображений: {count}")
        # Изменяем цвет текста в зависимости от количества
        if count < 2:
            self.image_count_label.config(fg="red")
        else:
            self.image_count_label.config(fg="green")

    def start_analysis(self):
        """Запуск анализа с проверкой"""
        if len(self.image_paths) < 2:
            msg = f"Загружено {len(self.image_paths)} изображений. Требуется минимум 2."
            logger.error(msg)
            messagebox.showerror("Ошибка", msg, parent=self.root)
            return

        try:
            size = (int(self.width_var.get()), int(self.height_var.get()))
            noise_level = float(self.noise_scale.get())
            logger.info(f"Начало анализа. Размер: {size}, Шум: {noise_level}")

            threading.Thread(
                target=self.run_analysis,
                args=(size, noise_level),
                daemon=True
            ).start()

        except ValueError as e:
            logger.error(f"Ошибка параметров: {e}")
            messagebox.showerror("Ошибка", str(e), parent=self.root)

    def run_analysis(self, size, noise_level):
        """Логика анализа с обработкой ошибок"""
        try:
            logger.debug(f"Запуск анализа в потоке. Изображений: {len(self.image_paths)}")

            output_pdf = "analysis_report.pdf"
            self.report_path.set(os.path.abspath(output_pdf))

            def callback(message):
                self.matrix_text.insert(tk.END, message + "\n")
                self.matrix_text.see(tk.END)
                self.root.update()

            results = process_images_and_generate_report(
                self.image_paths,
                output_pdf,
                size,
                noise_scale=noise_level,
                progress_callback=callback
            )

            self.update_plot(results)
            logger.info("Анализ успешно завершен")

            if os.path.exists(output_pdf):
                msg = f"Отчет сохранен:\n{os.path.abspath(output_pdf)}"
                logger.info(msg)
                messagebox.showinfo("Готово", msg)
            else:
                logger.error("Файл отчета не создан")
                messagebox.showerror("Ошибка", "Не удалось создать отчет")

        except Exception as e:
            logger.critical(f"Ошибка анализа: {str(e)}", exc_info=True)
            messagebox.showerror("Ошибка", str(e))

    def update_plot(self, results):
        """Обновление графика"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if 'statistics' in results and 'snr_error_data' in results['statistics']:
            # Извлечение данных для графика
            snr = [x[0] for x in results['statistics']['snr_error_data']]
            errors = [x[1] for x in results['statistics']['snr_error_data']]

            # Преобразование ошибок в числовой формат
            errors = [1 if e else 0 for e in errors]

            # Создание графика
            ax.scatter(snr, errors, alpha=0.5)
            ax.set_xlabel('SNR (дБ)')
            ax.set_ylabel('Факт ошибки (1-ошибка, 0-корректно)')
            ax.set_title('Зависимость ошибок от SNR')
            ax.grid(True)

            # Линия тренда
            if len(snr) > 1:
                z = np.polyfit(snr, errors, 1)
                p = np.poly1d(z)
                ax.plot(snr, p(snr), "r--")

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg
import threading
import os
from utils import load_image_as_matrix_and_vector, format_matrix, format_vector
from main_logic import process_images_and_generate_report


class App:
    def __init__(self, root):
        self.root = root
        root.title("Анализ качества распознавания")
        root.geometry("1200x800")

        # Валидация ввода
        self.validate_cmd = root.register(self.validate_size)

        # Панель управления
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10, fill=tk.X)

        # Поля ввода размеров
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

        # Кнопки
        tk.Button(control_frame, text="Добавить изображения", command=self.add_images).grid(row=0, column=4, padx=5)
        tk.Button(control_frame, text="Анализировать", command=self.start_analysis).grid(row=0, column=5)

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

    def validate_size(self, value):
        """Валидация ввода размеров"""
        if not value:
            return True
        try:
            num = int(value)
            return 3 <= num <= 100
        except ValueError:
            return False

    def add_images(self):
        """Добавление изображений"""
        files = filedialog.askopenfilenames(filetypes=[("Изображения", "*.png;*.jpg;*.bmp")])
        for path in files:
            if path not in self.image_paths:
                try:
                    # Загрузка миниатюры
                    img = Image.open(path).convert('RGB').resize((80, 112))
                    tk_img = ImageTk.PhotoImage(img)
                    lbl = tk.Label(self.thumb_frame, image=tk_img)
                    lbl.image = tk_img
                    lbl.pack(side="left", padx=5)
                    self.thumbnails.append(lbl)
                    self.image_paths.append(path)

                    # Загрузка матрицы
                    size = (int(self.width_var.get()), int(self.height_var.get()))
                    matrix, vector = load_image_as_matrix_and_vector(path, size)

                    # Вывод в текстовое поле
                    self.matrix_text.insert(tk.END, f"Изображение {len(self.image_paths)} ({size[0]}x{size[1]}):\n")
                    self.matrix_text.insert(tk.END, f"Матрица:\n{format_matrix(matrix)}\nВектор:\n")

                    # Подсветка единиц
                    vector_str = format_vector(vector)
                    for line in vector_str.split('\n'):
                        if line.strip() == '1':
                            self.matrix_text.insert(tk.END, line + '\n', 'highlight')
                        else:
                            self.matrix_text.insert(tk.END, line + '\n')

                    self.matrix_text.insert(tk.END, "=" * 50 + "\n")

                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить {path}:\n{str(e)}")

    def start_analysis(self):
        """Запуск анализа"""
        if len(self.image_paths) < 2:
            messagebox.showerror("Ошибка", "Необходимо минимум 2 изображения")
            return

        try:
            size = (int(self.width_var.get()), int(self.height_var.get()))
            if size[0] < 3 or size[1] < 3:
                raise ValueError("Минимальный размер 3x3 пикселя")

            threading.Thread(
                target=self.run_analysis,
                args=(size,),
                daemon=True
            ).start()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def run_analysis(self, size):
        """Основная логика анализа"""
        try:
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
                progress_callback=callback
            )

            self.update_plot(results)

            if os.path.exists(output_pdf):
                messagebox.showinfo("Готово", f"Отчет сохранен:\n{os.path.abspath(output_pdf)}")
            else:
                messagebox.showerror("Ошибка", "Не удалось создать отчет")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def update_plot(self, results):
        """Обновление графика"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if 'snr_vs_error' in results and results['snr_vs_error']:
            snr = [x[0] for x in results['snr_vs_error']]
            errors = [x[1] for x in results['snr_vs_error']]
            ax.scatter(snr, errors)
            ax.set_xlabel('SNR (дБ)')
            ax.set_ylabel('Вероятность ошибки')
            ax.set_title('Зависимость ошибок от SNR')
            ax.grid(True)

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
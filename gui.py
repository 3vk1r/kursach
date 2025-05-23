import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from PIL import Image, ImageTk
from main_logic import process_images_and_generate_report
import threading


class App:
    def __init__(self, root):
        self.root = root
        root.title("Контроль качества символов")
        root.geometry("1000x800")

        # Верхняя панель параметров
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10, fill=tk.X)

        # Поля для параметров
        tk.Label(control_frame, text="Ширина:").grid(row=0, column=0)
        self.width_var = tk.Entry(control_frame, width=5)
        self.width_var.grid(row=0, column=1)
        self.width_var.insert(0, "5")

        tk.Label(control_frame, text="Высота:").grid(row=0, column=2)
        self.height_var = tk.Entry(control_frame, width=5)
        self.height_var.grid(row=0, column=3)
        self.height_var.insert(0, "7")

        tk.Label(control_frame, text="Коэф. шума:").grid(row=0, column=4)
        self.scale_var = tk.Entry(control_frame, width=5)
        self.scale_var.grid(row=0, column=5)
        self.scale_var.insert(0, "3.0")

        # Кнопки
        tk.Button(control_frame, text="Добавить изображения", command=self.add_images).grid(row=0, column=6, padx=5)
        tk.Button(control_frame, text="Запустить анализ", command=self.start_analysis).grid(row=0, column=7)

        # Область вывода
        self.output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20)
        self.output_text.pack(pady=10, fill=tk.BOTH, expand=True)

        # Нижняя панель с миниатюрами
        self.thumb_frame = tk.Frame(root)
        self.thumb_frame.pack(pady=10, fill=tk.X)

        self.image_paths = []
        self.thumbnails = []

    def add_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Изображения", "*.png;*.jpg;*.bmp")])
        for path in files:
            if path not in self.image_paths:
                self.image_paths.append(path)
                self.add_thumbnail(path)
                self.log_message(f"Добавлено изображение: {path}")

    def add_thumbnail(self, path):
        img = Image.open(path).convert("RGB").resize((80, 112))
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(self.thumb_frame, image=tk_img)
        lbl.image = tk_img
        lbl.pack(side="left", padx=5)
        self.thumbnails.append(lbl)

    def start_analysis(self):
        if len(self.image_paths) < 2:
            messagebox.showerror("Ошибка", "Нужно минимум два изображения")
            return

        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            scale = float(self.scale_var.get())

            self.log_message("\n=== Начало анализа ===")
            self.log_message(f"Параметры: размер {width}x{height}, коэффициент шума {scale}")

            # Запуск в отдельном потоке чтобы GUI не зависал
            threading.Thread(
                target=self.run_analysis,
                args=(width, height, scale),
                daemon=True
            ).start()

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные параметры: {e}")

    def run_analysis(self, width, height, scale):
        try:
            # Перенаправляем вывод в GUI
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            process_images_and_generate_report(
                self.image_paths,
                scale,
                "report.pdf",
                (width, height))

            # Возвращаем вывод и показываем в GUI
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            self.log_message(output)
            self.log_message("\n=== Анализ завершён ===")
            self.log_message("PDF-отчёт сохранён как report.pdf")

            messagebox.showinfo("Готово", "Анализ завершён успешно!")

        except Exception as e:
            self.log_message(f"\nОШИБКА: {str(e)}")
            messagebox.showerror("Ошибка", str(e))

    def log_message(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
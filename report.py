from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Регистрируем шрифты с fallback
try:
    pdfmetrics.registerFont(TTFont("Arial", "arial.ttf"))
except:
    try:
        pdfmetrics.registerFont(TTFont("Arial", "DejaVuSans.ttf"))  # Для Linux
    except:
        pdfmetrics.registerFont(TTFont("Arial", "Helvetica"))  # Последний fallback


def create_pdf_report(output_path, img_paths, angles, snr_values, error_rate, size):
    pdf = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # Заголовок
    pdf.setFont("Arial", 16)
    pdf.drawCentredString(width / 2, height - 2 * cm, "Отчёт по оценке качества распознавания символов")

    # Параметры
    pdf.setFont("Arial", 12)
    y_position = height - 3.5 * cm
    pdf.drawString(2 * cm, y_position, f"Размер символов: {size[0]}x{size[1]} пикселей")
    y_position -= 0.7 * cm
    pdf.drawString(2 * cm, y_position, f"Коэффициент шума: {np.mean(snr_values):.2f} дБ")
    y_position -= 0.7 * cm
    pdf.drawString(2 * cm, y_position, f"Средняя вероятность ошибки: {error_rate:.2%}")

    # Изображения с углами
    y_position -= 1.5 * cm
    temp_files = []

    try:
        for i in range(len(img_paths) - 1):
            if y_position < 5 * cm:
                pdf.showPage()
                y_position = height - 2 * cm

            # Подготовка изображений
            img1_path = f"temp_img_{i}_1.jpg"
            img2_path = f"temp_img_{i}_2.jpg"

            for img_idx, path in enumerate([img_paths[i], img_paths[i + 1]]):
                try:
                    img = Image.open(path)
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img = img.resize((100, 140))
                    temp_path = img1_path if img_idx == 0 else img2_path
                    img.save(temp_path, quality=95)
                    temp_files.append(temp_path)
                except Exception as e:
                    print(f"Ошибка обработки {path}: {e}")
                    continue

            # Добавление в PDF
            pdf.drawImage(img1_path, 2 * cm, y_position - 5 * cm, width=3 * cm, height=4.2 * cm)
            pdf.drawImage(img2_path, 6 * cm, y_position - 5 * cm, width=3 * cm, height=4.2 * cm)
            pdf.drawString(10 * cm, y_position - 3 * cm, f"Угол между символами: {angles[i]:.2f}°")
            y_position -= 6 * cm

        # График SNR
        pdf.showPage()
        snr_plot = "snr_plot.png"
        plt.figure(figsize=(10, 5))
        plt.plot(snr_values, label='SNR по итерациям')
        plt.xlabel("Номер итерации")
        plt.ylabel("SNR (дБ)")
        plt.title("Зависимость отношения сигнал-шум")
        plt.grid(True)
        plt.savefig(snr_plot, dpi=100, bbox_inches='tight')
        plt.close()

        pdf.drawImage(snr_plot, 2 * cm, height - 12 * cm, width=16 * cm, height=8 * cm)
        temp_files.append(snr_plot)

    finally:
        # Удаление временных файлов
        for file in temp_files:
            try:
                os.remove(file)
            except:
                pass

    pdf.save()
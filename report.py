from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import numpy as np
import os

def create_pdf_report(output_path, img_paths, results, size):
    """Создает PDF-отчет с учетом динамических размеров"""
    try:
        pdf = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4

        # Установка шрифта
        try:
            pdfmetrics.registerFont(TTFont("Courier", "cour.ttf"))
        except:
            pdfmetrics.registerFont(TTFont("Courier", "Courier-Bold"))

        # Заголовок
        pdf.setFont("Courier", 16)
        pdf.drawCentredString(width/2, height-2*cm, "Отчет по анализу символов")

        # Основные параметры
        pdf.setFont("Courier", 12)
        y = height - 3.5*cm
        pdf.drawString(2*cm, y, f"Проанализировано изображений: {len(img_paths)}")
        y -= 0.7*cm
        pdf.drawString(2*cm, y, f"Использованный размер: {size[0]}x{size[1]}")
        y -= 0.7*cm
        pdf.drawString(2*cm, y, f"Средняя вероятность ошибки: {np.mean(results['error_rates']):.2%}")

        # Матрицы изображений
        pdf.showPage()
        pdf.setFont("Courier", 14)
        pdf.drawString(2*cm, height-2*cm, f"Матрицы изображений ({size[0]}x{size[1]}):")

        y = height - 3*cm
        for i, matrix in enumerate(results['matrices'][:3]):
            pdf.setFont("Courier", 10)
            pdf.drawString(2*cm, y, f"Изображение {i+1}:")
            y -= 0.7*cm

            for row in matrix:
                pdf.drawString(2*cm, y, " ".join(str(int(x)) for x in row))
                y -= 0.5*cm

            y -= 1*cm
            if y < 3*cm and i < 2:
                pdf.showPage()
                y = height - 2*cm

        # График SNR vs Error
        pdf.showPage()
        pdf.setFont("Courier", 16)
        pdf.drawString(2*cm, height-2*cm, "Зависимость ошибок от SNR")

        plt.figure(figsize=(8, 4))
        snr = [x[0] for x in results['snr_vs_error']]
        errors = [x[1] for x in results['snr_vs_error']]
        plt.scatter(snr, errors)
        plt.xlabel("SNR (дБ)")
        plt.ylabel("Вероятность ошибки")
        plt.grid(True)

        graph_path = "temp_plot.png"
        plt.savefig(graph_path, dpi=100, bbox_inches='tight')
        plt.close()

        pdf.drawImage(graph_path, 2*cm, height-12*cm, width=14*cm, height=7*cm)
        os.remove(graph_path)

        pdf.save()

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Ошибка при создании отчета: {str(e)}")
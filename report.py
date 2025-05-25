import logging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import numpy as np
import os

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_pdf_report(output_path, img_paths, results, size):
    """Создает PDF-отчет с защитой от ошибок форматирования"""
    try:
        logger.info(f"Начало создания отчета. Путь: {output_path}")
        pdf = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4

        # Установка шрифта
        try:
            pdfmetrics.registerFont(TTFont("Courier", "cour.ttf"))
            logger.debug("Шрифт Courier зарегистрирован")
        except Exception as e:
            logger.warning(f"Не удалось загрузить шрифт Courier: {e}")
            pdfmetrics.registerFont(TTFont("Courier", "Courier-Bold"))
            logger.debug("Использован резервный шрифт Courier-Bold")

        # Заголовок
        pdf.setFont("Courier", 16)
        pdf.drawCentredString(width / 2, height - 2 * cm, "Отчет по анализу символов")

        # Основные параметры
        pdf.setFont("Courier", 12)
        y = height - 3.5 * cm
        pdf.drawString(2 * cm, y, f"Проанализировано изображений: {len(img_paths)}")
        y -= 0.7 * cm
        pdf.drawString(2 * cm, y, f"Использованный размер: {size[0]}x{size[1]}")
        y -= 0.7 * cm
        pdf.drawString(2 * cm, y, f"Средняя вероятность ошибки: {np.mean(results['error_rates']):.2%}")

        # Матрицы изображений
        logger.debug("Начало обработки матриц изображений")
        pdf.showPage()
        pdf.setFont("Courier", 14)
        pdf.drawString(2 * cm, height - 2 * cm, f"Матрицы изображений ({size[0]}x{size[1]}):")

        y = height - 3 * cm
        for i, matrix in enumerate(results['matrices'][:3]):
            try:
                logger.debug(f"Обработка матрицы {i + 1}")
                pdf.setFont("Courier", 10)
                pdf.drawString(2 * cm, y, f"Изображение {i + 1}:")
                y -= 0.7 * cm

                # Преобразуем матрицу к единому формату
                if isinstance(matrix, np.ndarray):
                    matrix = matrix.tolist()
                elif not isinstance(matrix, list):
                    matrix = [list(row) for row in matrix]

                logger.debug(f"Тип matrix после преобразования: {type(matrix)}")

                # Форматируем каждую строку
                for row in matrix:
                    try:
                        # Преобразуем строку в список чисел, если это необходимо
                        if not isinstance(row, list):
                            row = list(row)

                        # Форматируем элементы строки
                        row_str = " ".join([str(int(float(x))) for x in row])
                        pdf.drawString(2 * cm, y, row_str)
                        y -= 0.5 * cm

                    except Exception as e:
                        logger.error(f"Ошибка форматирования строки: {e}")
                        logger.error(f"Содержимое строки: {row}")
                        raise ValueError(f"Ошибка форматирования строки матрицы: {e}")

                y -= 1 * cm
                if y < 3 * cm and i < 2:
                    pdf.showPage()
                    y = height - 2 * cm

            except Exception as e:
                logger.error(f"Ошибка обработки матрицы {i + 1}: {e}")
                raise

        # График SNR vs Error
        logger.debug("Создание графика SNR vs Error")
        pdf.showPage()
        pdf.setFont("Courier", 16)
        pdf.drawString(2 * cm, height - 2 * cm, "Зависимость ошибок от SNR")

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

        pdf.drawImage(graph_path, 2 * cm, height - 12 * cm, width=14 * cm, height=7 * cm)
        os.remove(graph_path)

        pdf.save()
        logger.info("Отчет успешно создан")


    except Exception as e:

        logger.error(f"Критическая ошибка при создании отчета: {e}")

        if os.path.exists(output_path):
            os.remove(output_path)

        raise RuntimeError(f"Ошибка при создании отчета: {str(e)}")
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from PIL import Image

# Настройка логирования
logger = logging.getLogger(__name__)


def create_pdf_report(output_path, img_paths, results, size):
    """Создает PDF-отчет с учетом всех исправлений"""
    try:
        logger.info(f"Начало создания отчета: {output_path}")

        # Инициализация документа
        pdf = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        margin = 2 * cm

        # Регистрация шрифтов
        font_name, bold_font_name = _register_fonts()

        # Титульная страница
        _add_title_page(pdf, width, height, margin, font_name, results, img_paths, size)

        # Страница с миниатюрами
        pdf.showPage()
        _add_thumbnails_page(pdf, img_paths, width, height, margin, font_name)

        # Страницы попарного анализа
        if 'pairwise_analysis' in results:
            for pair in results['pairwise_analysis']:
                pdf.showPage()
                _add_pair_analysis_page(pdf, pair, width, height, margin, font_name, bold_font_name)

        # Статистические страницы
        pdf.showPage()
        _add_statistics_page(pdf, results, width, height, margin, font_name)

        # Сохранение PDF
        pdf.save()
        logger.info(f"Отчет успешно создан: {output_path}")

    except Exception as e:
        logger.critical(f"Ошибка создания отчета: {str(e)}", exc_info=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Ошибка при создании отчета: {str(e)}")


def _register_fonts():
    """Регистрирует шрифты и возвращает имена основного и жирного шрифтов"""
    try:
        # Попытка зарегистрировать DejaVu Sans
        pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"))
        return "DejaVuSans", "DejaVuSans-Bold"
    except Exception as e:
        logger.warning(f"Не удалось зарегистрировать DejaVuSans: {str(e)}. Использую стандартные шрифты.")

    try:
        # Стандартные PDF-шрифты
        pdfmetrics.registerFont(TTFont("Helvetica", "Helvetica"))
        pdfmetrics.registerFont(TTFont("Helvetica-Bold", "Helvetica-Bold"))
        return "Helvetica", "Helvetica-Bold"
    except Exception as e:
        logger.error(f"Ошибка регистрации шрифтов: {str(e)}")
        raise RuntimeError("Не удалось зарегистрировать необходимые шрифты")


def _add_title_page(pdf, width, height, margin, font_name, results, img_paths, size):
    """Добавляет титульную страницу"""
    pdf.setFont(font_name, 16)
    pdf.drawCentredString(width / 2, height - margin, "Аналитический отчет")

    # Параметры анализа
    params = results.get('input_parameters', {
        'image_count': len(img_paths),
        'image_size': size,
        'noise_level': 'N/A',
        'analysis_date': 'N/A'
    })

    y = height - margin - 2 * cm
    pdf.setFont(font_name, 12)
    _draw_text_line(pdf, margin, y, f"Дата анализа: {params.get('analysis_date', 'N/A')}")
    y -= 0.7 * cm
    _draw_text_line(pdf, margin, y, f"Количество изображений: {params['image_count']}")
    y -= 0.7 * cm
    _draw_text_line(pdf, margin, y, f"Размер матриц: {params['image_size'][0]}x{params['image_size'][1]}")
    y -= 0.7 * cm
    _draw_text_line(pdf, margin, y, f"Уровень шума: {params.get('noise_level', 'N/A'):.1f}")


def _draw_text_line(pdf, x, y, text):
    """Вспомогательная функция для отрисовки текста"""
    pdf.drawString(x, y, text)


def _add_thumbnails_page(pdf, img_paths, width, height, margin, font_name):
    """Добавляет страницу с миниатюрами изображений"""
    pdf.setFont(font_name, 16)
    pdf.drawString(margin, height - margin, "Исходные изображения")

    thumb_size = 5 * cm
    x_positions = [margin, width / 2 - thumb_size / 2, width - margin - thumb_size]
    y_start = height - margin - 2 * cm
    current_y = y_start

    for i, path in enumerate(img_paths):
        try:
            col = i % 3
            row = i // 3

            if current_y < margin + thumb_size:
                pdf.showPage()
                current_y = y_start
                pdf.setFont(font_name, 16)
                pdf.drawString(margin, height - margin, "Исходные изображения (продолжение)")

            with Image.open(path) as img:
                # Конвертация RGBA в RGB при необходимости
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.thumbnail((thumb_size, thumb_size))
                temp_path = f"temp_{i}.jpg"
                img.save(temp_path)
                pdf.drawImage(temp_path, x_positions[col], current_y - thumb_size,
                              width=thumb_size, height=thumb_size)
                os.remove(temp_path)

                pdf.setFont(font_name, 8)
                pdf.drawString(x_positions[col], current_y - thumb_size - 0.5 * cm,
                               f"Изобр. {i + 1}: {os.path.basename(path)}")

            if (i + 1) % 3 == 0:
                current_y -= thumb_size + 1 * cm

        except Exception as e:
            logger.error(f"Ошибка добавления миниатюры {path}: {str(e)}")
            continue


def _add_pair_analysis_page(pdf, pair, width, height, margin, font_name, bold_font_name):
    """Добавляет страницу анализа для пары изображений"""
    pdf.setFont(font_name, 16)
    pdf.drawString(margin, height - margin, f"Анализ пары: {pair.get('pair_id', 'N/A')}")

    # Добавление изображений
    img_size = 8 * cm
    try:
        if 'images' in pair:
            _draw_image(pdf, pair['images'].get('source'), margin, height - margin - 2 * cm - img_size, img_size)
            _draw_image(pdf, pair['images'].get('target'), width - margin - img_size,
                        height - margin - 2 * cm - img_size, img_size)
    except Exception as e:
        logger.error(f"Ошибка загрузки изображений пары: {str(e)}")

    # Информация о паре
    pdf.setFont(font_name, 12)
    info_y = height - margin - 2.5 * cm - img_size
    _draw_text_line(pdf, margin, info_y, f"Угол между векторами: {pair.get('vector_angle', 'N/A'):.2f}°")

    # Таблица экспериментов
    if 'experiments' in pair:
        _draw_experiments_table(pdf, pair['experiments'], margin, info_y - 3 * cm, font_name, bold_font_name, width,
                                height)


def _draw_image(pdf, path, x, y, size):
    """Отрисовывает изображение с указанными параметрами"""
    if path and os.path.exists(path):
        with Image.open(path) as img:
            # Конвертация RGBA в RGB при необходимости
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.thumbnail((size, size))
            temp_path = "temp_img.jpg"
            img.save(temp_path)
            pdf.drawImage(temp_path, x, y, width=size, height=size)
            os.remove(temp_path)


def _draw_experiments_table(pdf, experiments, x, y, font_name, bold_font_name, width, height):
    """Отрисовывает таблицу с результатами экспериментов"""
    # Проверка доступности жирного шрифта
    try:
        pdfmetrics.getFont(bold_font_name)
        header_font = bold_font_name
    except:
        header_font = font_name
        logger.warning(f"Жирный шрифт {bold_font_name} недоступен, использую {font_name}")

    table_data = [["Эксп.", "Шум", "SNR (дБ)", "Ошибка"]]
    for exp in experiments:
        table_data.append([
            str(exp.get('experiment_id', '')),
            f"{exp.get('noise_level', 0):.2f}",
            f"{exp.get('snr', 0):.1f}",
            "Да" if exp.get('error', False) else "Нет"
        ])

    table = Table(table_data, colWidths=[2 * cm, 3 * cm, 4 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), header_font),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    table.wrapOn(pdf, width, height)
    table.drawOn(pdf, x, y)


def _add_statistics_page(pdf, results, width, height, margin, font_name):
    """Добавляет страницу со статистикой"""
    pdf.setFont(font_name, 16)
    pdf.drawString(margin, height - margin, "Статистические данные")

    # График SNR vs Ошибки
    if 'statistics' in results:
        _draw_snr_plot(pdf, results['statistics'], width, height, margin)
        _draw_angles_histogram(pdf, results['statistics'], width, height, margin)


def _draw_snr_plot(pdf, stats, width, height, margin):
    """Отрисовывает график зависимости SNR от ошибок"""
    if 'snr_values' in stats and 'error_rates' in stats:
        try:
            plt.figure(figsize=(10, 5))
            plt.scatter(stats['snr_values'], stats['error_rates'], alpha=0.5)
            plt.xlabel("SNR (дБ)")
            plt.ylabel("Частота ошибок")
            plt.grid(True)
            plot_path = "temp_snr_plot.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            pdf.drawImage(plot_path, margin, height - margin - 6 * cm, width=width - 2 * margin, height=5 * cm)
            os.remove(plot_path)
        except Exception as e:
            logger.error(f"Ошибка создания графика SNR: {str(e)}")


def _draw_angles_histogram(pdf, stats, width, height, margin):
    """Отрисовывает гистограмму углов"""
    if 'angles' in stats:
        try:
            plt.figure(figsize=(10, 5))
            plt.hist(stats['angles'], bins=15, color='skyblue', edgecolor='black')
            plt.xlabel("Угол между векторами (°)")
            plt.ylabel("Количество пар")
            plt.grid(True)
            hist_path = "temp_hist.png"
            plt.savefig(hist_path, dpi=100, bbox_inches='tight')
            plt.close()
            pdf.drawImage(hist_path, margin, height - margin - 13 * cm, width=width - 2 * margin, height=5 * cm)
            os.remove(hist_path)
        except Exception as e:
            logger.error(f"Ошибка создания гистограммы: {str(e)}")
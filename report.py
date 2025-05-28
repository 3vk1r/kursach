from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from PIL import Image
from scipy.stats import linregress

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
        if 'statistics' in results:
            # График SNR vs Ошибки
            pdf.showPage()
            pdf.setPageSize(landscape(A4))
            _add_snr_plot_page(pdf, results, landscape(A4)[0], landscape(A4)[1], margin, font_name)

            # Гистограмма углов
            pdf.showPage()
            pdf.setPageSize(landscape(A4))
            _add_angles_histogram_page(pdf, results, landscape(A4)[0], landscape(A4)[1], margin, font_name)

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
        if 'image1' in pair and 'image2' in pair:
            logger.info(f"Добавление изображений для пары {pair['pair_id']}: "
                        f"{pair['image1']['path']} и {pair['image2']['path']}")

            # Отрисовка первого изображения
            _draw_image(pdf, pair['image1']['path'], margin, height - margin - 2 * cm - img_size, img_size)
            pdf.setFont(font_name, 10)
            pdf.drawString(margin, height - margin - 2 * cm - img_size - 0.5 * cm,
                           f"Изобр. {pair['image1']['index']}: {pair['image1']['name']}")

            # Отрисовка второго изображения
            _draw_image(pdf, pair['image2']['path'], width - margin - img_size,
                        height - margin - 2 * cm - img_size, img_size)
            pdf.setFont(font_name, 10)
            pdf.drawString(width - margin - img_size, height - margin - 2 * cm - img_size - 0.5 * cm,
                           f"Изобр. {pair['image2']['index']}: {pair['image2']['name']}")
    except Exception as e:
        logger.error(f"Ошибка загрузки изображений пары: {str(e)}")
        pdf.drawString(margin, height - margin - 2 * cm, f"Ошибка: {str(e)}")
    # Информация о паре
    pdf.setFont(font_name, 12)
    info_y = height - margin - 2.5 * cm - img_size - 1 * cm
    _draw_text_line(pdf, margin, info_y, f"Угол между векторами: {pair.get('vector_angle', 'N/A'):.2f}°")
    info_y -= 0.7 * cm
    _draw_text_line(pdf, margin, info_y, f"Вероятность ошибки: {pair.get('error_rate', 'N/A'):.2%}")
    info_y -= 0.7 * cm
    _draw_text_line(pdf, margin, info_y, f"Средний SNR: {pair.get('avg_snr', 'N/A'):.2f} дБ")


def _draw_image(pdf, path, x, y, size):
    """Отрисовывает изображение с указанными параметрами"""
    if path and os.path.exists(path):
        try:
            with Image.open(path) as img:
                # Сохраняем оригинальное изображение во временный файл
                original_temp_path = f"original_temp_{os.path.basename(path)}"
                img.save(original_temp_path)

                # Создаем миниатюру
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.thumbnail((size, size))

                # Сохраняем миниатюру во временный файл
                temp_path = f"temp_{os.path.basename(path)}"
                img.save(temp_path)

                # Рисуем миниатюру в PDF
                pdf.drawImage(temp_path, x, y, width=size, height=size)

                # Удаляем временные файлы
                os.remove(original_temp_path)
                os.remove(temp_path)

                logger.debug(f"Изображение {path} успешно отрисовано")
        except Exception as e:
            logger.error(f"Ошибка отрисовки изображения {path}: {str(e)}")


def _add_snr_plot_page(pdf, results, width, height, margin, font_name):
    """Добавляет отдельную страницу для графика SNR в альбомной ориентации"""
    pdf.setFont(font_name, 16)
    pdf.drawCentredString(width / 2, height - margin, "Зависимость вероятности ошибки от SNR")

    if 'statistics' in results and 'snr_error_data' in results['statistics']:
        _draw_snr_plot(pdf, results['statistics'], width, height, margin)


def _add_angles_histogram_page(pdf, results, width, height, margin, font_name):
    """Добавляет отдельную страницу для гистограммы углов в альбомной ориентации"""
    pdf.setFont(font_name, 16)
    pdf.drawCentredString(width / 2, height - margin, "Распределение углов между векторами")

    if 'statistics' in results and 'angles' in results['statistics']:
        _draw_angles_histogram(pdf, results['statistics'], width, height, margin)


def _draw_snr_plot(pdf, stats, width, height, margin):
    """Отрисовывает график зависимости вероятности ошибки от SNR"""
    if 'snr_error_data' in stats and stats['snr_error_data']:
        try:
            # Подготовка данных
            snr_values = [x[0] for x in stats['snr_error_data']]
            errors = [x[1] for x in stats['snr_error_data']]

            # Создаем биннинг для SNR
            min_snr = min(snr_values)
            max_snr = max(snr_values)
            bin_edges = np.linspace(min_snr, max_snr, 11)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Вычисляем вероятность ошибки для каждого бина
            bin_errors = []
            for i in range(len(bin_edges) - 1):
                lower = bin_edges[i]
                upper = bin_edges[i + 1]
                bin_data = [e for s, e in zip(snr_values, errors) if lower <= s < upper]

                if bin_data:
                    bin_errors.append(np.mean(bin_data))
                else:
                    bin_errors.append(0)

            plt.figure(figsize=(12, 8))

            # Основной график
            plt.plot(bin_centers, bin_errors, 'o-', linewidth=2, markersize=8, color='blue')

            # Настройки графика
            plt.xlabel("Отношение сигнал-шум (SNR), дБ", fontsize=12)
            plt.ylabel("Вероятность ошибки", fontsize=12)
            plt.title("Зависимость вероятности ошибки от SNR", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)

            # Сохранение графика
            plot_path = "temp_snr_plot.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Размещение графика по центру страницы
            pdf.drawImage(plot_path,
                          margin,
                          margin,
                          width=width - 2 * margin,
                          height=height - 2 * margin,
                          preserveAspectRatio=True,
                          anchor='c')
            os.remove(plot_path)
        except Exception as e:
            logger.error(f"Ошибка создания графика SNR: {str(e)}")


def _draw_angles_histogram(pdf, stats, width, height, margin):
    """Отрисовывает гистограмму углов"""
    if 'angles' in stats:
        try:
            plt.figure(figsize=(12, 8))

            # Гистограмма с 15 бинами
            plt.hist(stats['angles'], bins=15, color='skyblue', edgecolor='black')

            # Настройки графика
            plt.xlabel("Угол между векторами, °", fontsize=12)
            plt.ylabel("Количество пар", fontsize=12)
            plt.title("Распределение углов между векторами", fontsize=14)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)

            # Сохранение гистограммы
            hist_path = "temp_hist.png"
            plt.savefig(hist_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Размещение гистограммы по центру страницы
            pdf.drawImage(hist_path,
                          margin,
                          margin,
                          width=width - 2 * margin,
                          height=height - 2 * margin,
                          preserveAspectRatio=True,
                          anchor='c')
            os.remove(hist_path)
        except Exception as e:
            logger.error(f"Ошибка создания гистограммы: {str(e)}")
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import numpy as np
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import shutil

# Регистрация кириллического шрифта
try:
    pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', 'arialbd.ttf'))
except:
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
        pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'DejaVuSans-Bold.ttf'))
    except:
        pass

# Настройка Matplotlib для работы без дисплея
plt.switch_backend('Agg')

# Настройка логирования
logger = logging.getLogger(__name__)


def create_pdf_report(output_path, img_paths, results, size):
    temp_dir = tempfile.mkdtemp()
    try:
        logger.info(f"Создание отчёта: {output_path}")
        pdf = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        margin = 1.5 * cm

        # Используем кириллический шрифт по умолчанию
        pdf.setFont("Arial", 12)

        # Титульная страница
        _add_title_page(pdf, width, height, margin, results)

        # Страница с миниатюрами изображений
        pdf.showPage()
        _add_thumbnails_page(pdf, img_paths, width, height, margin, temp_dir)

        # Страница с матрицами
        pdf.showPage()
        _add_matrices_page(pdf, results, width, height, margin)

        # Страница с графиком собственных значений
        pdf.showPage()
        _add_eigenvalues_page(pdf, results, width, height, margin, temp_dir)

        # Страница с углами
        pdf.showPage()
        _add_angles_page(pdf, results, width, height, margin)

        # Страница с графиком углов и миниатюрами
        pdf.showPage()
        _add_angles_plot_page(pdf, results, width, height, margin, temp_dir, img_paths)

        pdf.save()
        logger.info(f"Отчёт создан: {output_path}")
    except Exception as e:
        logger.critical(f"Ошибка создания отчёта: {str(e)}", exc_info=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _add_title_page(pdf, width, height, margin, results):
    """Титульная страница"""
    # Используем кириллический шрифт
    pdf.setFont("Arial-Bold", 18)
    pdf.drawCentredString(width / 2, height - margin, "ОТЧЁТ ПО АНАЛИЗУ ИЗОБРАЖЕНИЙ")

    pdf.setFont("Arial", 12)
    y = height - margin - 2 * cm
    params = results['input_parameters']

    pdf.drawCentredString(width / 2, y, f"Дата анализа: {params['analysis_date']}")
    y -= 1 * cm
    pdf.drawCentredString(width / 2, y, f"Количество изображений: {params['image_count']}")
    y -= 1 * cm
    pdf.drawCentredString(width / 2, y, f"Размер матриц: {params['image_size'][0]}x{params['image_size'][1]}")

    # Линия разделитель
    pdf.line(margin, y - 1 * cm, width - margin, y - 1 * cm)

    # Пояснение
    y -= 2 * cm
    pdf.setFont("Arial", 10)
    pdf.drawString(margin, y, "Анализ включает:")
    y -= 0.5 * cm
    pdf.drawString(margin + 0.5 * cm, y, "• Загрузку и векторизацию изображений")
    y -= 0.5 * cm
    pdf.drawString(margin + 0.5 * cm, y, "• Расчет ковариационной матрицы")
    y -= 0.5 * cm
    pdf.drawString(margin + 0.5 * cm, y, "• Попарный анализ углов между векторами")
    y -= 0.5 * cm
    pdf.drawString(margin + 0.5 * cm, y, "• Решение систем уравнений методом наименьших квадратов")


def _add_thumbnails_page(pdf, img_paths, width, height, margin, temp_dir):
    """Страница с миниатюрами изображений"""
    pdf.setFont("Arial-Bold", 16)
    pdf.drawCentredString(width / 2, height - margin, "ИСХОДНЫЕ ИЗОБРАЖЕНИЯ")

    # Параметры миниатюр
    thumb_size = 5 * cm
    images_per_row = min(3, len(img_paths))
    spacing = 1 * cm

    # Рассчет позиций
    start_x = (width - (images_per_row * thumb_size + (images_per_row - 1) * spacing)) / 2
    start_y = height - margin - 2 * cm

    # Отображение изображений
    for i, path in enumerate(img_paths):
        row = i // images_per_row
        col = i % images_per_row

        x = start_x + col * (thumb_size + spacing)
        y = start_y - row * (thumb_size + spacing + 0.5 * cm)

        # Проверка, не вышли ли за пределы страницы
        if y < margin + 1 * cm:
            # Создаем новую страницу
            pdf.showPage()
            pdf.setFont("Arial-Bold", 16)
            pdf.drawCentredString(width / 2, height - margin, "ИЗОБРАЖЕНИЯ (ПРОДОЛЖЕНИЕ)")
            start_y = height - margin - 1 * cm
            row = 0
            y = start_y - row * (thumb_size + spacing + 0.5 * cm)
            x = start_x + (i % images_per_row) * (thumb_size + spacing)

        try:
            # Создаем временный файл для изображения
            temp_file = os.path.join(temp_dir, f"thumb_{i}.png")

            # Загрузка и обработка изображения
            with Image.open(path) as img:
                # Конвертируем в RGB, если нужно
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Создаем миниатюру с сохранением пропорций
                img.thumbnail((int(thumb_size), int(thumb_size)))
                img.save(temp_file, format='PNG')

            # Добавляем в PDF
            pdf.drawImage(temp_file, x, y - thumb_size,
                          width=thumb_size, height=thumb_size)

            # Подпись
            pdf.setFont("Arial", 8)
            pdf.drawCentredString(x + thumb_size / 2, y - thumb_size - 0.5 * cm,
                                  f"Изобр. {i + 1}: {os.path.basename(path)}")

        except Exception as e:
            logger.error(f"Ошибка добавления миниатюры {path}: {str(e)}")
            pdf.setFont("Arial", 8)
            pdf.drawString(x, y - thumb_size, f"Ошибка: {str(e)}")


def _add_matrices_page(pdf, results, width, height, margin):
    """Страница с матрицами изображений"""
    pdf.setFont("Arial-Bold", 16)
    pdf.drawCentredString(width / 2, height - margin, "МАТРИЦЫ ИЗОБРАЖЕНИЙ")

    # Используем моноширинный шрифт для матриц
    pdf.setFont("Courier", 8)
    y = height - margin - 1.5 * cm
    line_height = 0.4 * cm
    max_rows_per_page = 40

    row_count = 0
    for i, matrix in enumerate(results['matrices']):
        # Заголовок
        pdf.setFont("Arial-Bold", 10)
        pdf.drawString(margin, y, f"Изображение {i + 1}:")
        y -= line_height
        row_count += 1

        # Матрица
        pdf.setFont("Courier", 8)
        for row in matrix:
            # Преобразуем в строку с пробелами
            row_str = " ".join([str(int(x)) for x in row])
            pdf.drawString(margin + 0.5 * cm, y, row_str)
            y -= line_height
            row_count += 1

            # Проверка места на странице
            if row_count > max_rows_per_page:
                pdf.showPage()
                pdf.setFont("Arial-Bold", 16)
                pdf.drawCentredString(width / 2, height - margin, "МАТРИЦЫ (ПРОДОЛЖЕНИЕ)")
                y = height - margin - 1.5 * cm
                row_count = 0
                pdf.setFont("Courier", 8)

        y -= line_height  # Отступ между матрицами
        row_count += 1


def _add_eigenvalues_page(pdf, results, width, height, margin, temp_dir):
    """Страница с графиком собственных значений"""
    pdf.setFont("Arial-Bold", 16)
    pdf.drawCentredString(width / 2, height - margin, "СОБСТВЕННЫЕ ЗНАЧЕНИЯ КОВАРИАЦИОННОЙ МАТРИЦЫ")

    if 'cov_matrix' not in results or not results['cov_matrix']:
        pdf.setFont("Arial", 10)
        pdf.drawString(margin, height - margin - 1 * cm, "Матрица не вычислена")
        return

    # Создаем график собственных значений
    try:
        cov_matrix = np.array(results['cov_matrix'])

        # Вычисляем собственные значения
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        # Сортируем по убыванию
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Создаем график
        plt.figure(figsize=(10, 6))
        plt.plot(eigenvalues, 'bo-')
        plt.title("Собственные значения ковариационной матрицы")
        plt.xlabel("Номер компоненты")
        plt.ylabel("Собственное значение")
        plt.grid(True)
        plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации

        # Сохраняем во временный файл
        temp_file = os.path.join(temp_dir, "eigenvalues.png")
        plt.savefig(temp_file, format='png', dpi=150, bbox_inches='tight')
        plt.close()

        # Размещаем график
        pdf.drawImage(temp_file, margin, height / 3,
                      width=width - 2 * margin, height=height / 2.5,
                      preserveAspectRatio=True, anchor='c')

        # Добавляем статистику
        pdf.setFont("Arial", 10)
        y_pos = height / 3 - 2 * cm
        pdf.drawString(margin, y_pos, f"Количество компонент: {len(eigenvalues)}")
        y_pos -= 0.7 * cm
        pdf.drawString(margin, y_pos, f"Максимальное значение: {eigenvalues[0]:.4f}")
        y_pos -= 0.7 * cm
        pdf.drawString(margin, y_pos, f"Минимальное значение: {eigenvalues[-1]:.4f}")
        y_pos -= 0.7 * cm
        pdf.drawString(margin, y_pos, f"Сумма значений: {np.sum(eigenvalues):.4f}")

    except Exception as e:
        logger.error(f"Ошибка создания графика собственных значений: {str(e)}")
        pdf.setFont("Arial", 10)
        pdf.drawString(margin, height - margin - 1 * cm, f"Ошибка визуализации: {str(e)}")


def _add_angles_page(pdf, results, width, height, margin):
    """Страница с углами между векторами"""
    pdf.setFont("Arial-Bold", 16)
    pdf.drawCentredString(width / 2, height - margin, "УГЛЫ МЕЖДУ ВЕКТОРАМИ")

    if 'pairwise_analysis' not in results or not results['pairwise_analysis']:
        pdf.setFont("Arial", 10)
        pdf.drawString(margin, height - margin - 1 * cm, "Данные отсутствуют")
        return

    # Создаем таблицу
    pdf.setFont("Arial-Bold", 10)
    pdf.drawString(margin, height - margin - 1 * cm, "Пара изображений")
    pdf.drawString(width / 3, height - margin - 1 * cm, "Угол (°)")
    pdf.drawString(2 * width / 3, height - margin - 1 * cm, "Невязка")
    pdf.line(margin, height - margin - 1.2 * cm, width - margin, height - margin - 1.2 * cm)

    y = height - margin - 1.5 * cm
    line_height = 0.6 * cm

    for pair in results['pairwise_analysis']:
        # Проверка места на странице
        if y < margin + line_height:
            pdf.showPage()
            pdf.setFont("Arial-Bold", 16)
            pdf.drawCentredString(width / 2, height - margin, "УГЛЫ МЕЖДУ ВЕКТОРАМИ (ПРОДОЛЖЕНИЕ)")
            y = height - margin - 1.5 * cm
            pdf.setFont("Arial-Bold", 10)
            pdf.drawString(margin, y + 0.3 * cm, "Пара изображений")
            pdf.drawString(width / 3, y + 0.3 * cm, "Угол (°)")
            pdf.drawString(2 * width / 3, y + 0.3 * cm, "Невязка")
            pdf.line(margin, y + 0.1 * cm, width - margin, y + 0.1 * cm)
            y -= line_height

        angle = pair.get('vector_angle', 'N/A')
        residual = pair.get('residual', 'N/A')

        # Форматирование значений
        angle_str = f"{angle:.2f}°" if isinstance(angle, float) else str(angle)
        residual_str = f"{residual:.4f}" if isinstance(residual, float) else str(residual)

        # Рисуем строку таблицы
        pdf.setFont("Arial", 10)
        pdf.drawString(margin, y, f"Пара {pair['pair_id']}")
        pdf.drawString(width / 3, y, angle_str)
        pdf.drawString(2 * width / 3, y, residual_str)

        y -= line_height


def _add_angles_plot_page(pdf, results, width, height, margin, temp_dir, img_paths):
    """Страница с графиком распределения углов и миниатюрами для min/max углов"""
    pdf.setFont("Arial-Bold", 16)
    pdf.drawCentredString(width / 2, height - margin, "РАСПРЕДЕЛЕНИЕ УГЛОВ МЕЖДУ ВЕКТОРАМИ")

    if 'statistics' not in results or 'angles' not in results['statistics']:
        pdf.setFont("Arial", 10)
        pdf.drawString(margin, height - margin - 1 * cm, "Данные для графика отсутствуют")
        return

    angles = results['statistics']['angles']
    if not angles:
        pdf.setFont("Arial", 10)
        pdf.drawString(margin, height - margin - 1 * cm, "Нет данных об углах")
        return

    try:
        # Находим пары с минимальным и максимальным углом
        min_angle = float('inf')
        max_angle = float('-inf')
        min_pair = None
        max_pair = None

        for pair in results['pairwise_analysis']:
            angle = pair.get('vector_angle', None)
            if angle is not None:
                if angle < min_angle:
                    min_angle = angle
                    min_pair = pair
                if angle > max_angle:
                    max_angle = angle
                    max_pair = pair

        # Создаем гистограмму
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(angles, bins=15, color='skyblue', edgecolor='black')

        # Добавляем линию среднего значения
        mean_angle = np.mean(angles)
        plt.axvline(mean_angle, color='red', linestyle='dashed', linewidth=1)
        plt.text(mean_angle + 1, max(n) * 0.9, f'Среднее: {mean_angle:.1f}°', color='red')

        # Настройки оформления
        plt.title("Распределение углов между векторами изображений")
        plt.xlabel("Угол между векторами (°)")
        plt.ylabel("Количество пар")
        plt.grid(axis='y', alpha=0.75)

        # Сохраняем во временный файл
        temp_file = os.path.join(temp_dir, "angles_hist.png")
        plt.savefig(temp_file, format='png', dpi=150, bbox_inches='tight')
        plt.close()

        # Размещаем график (уменьшаем высоту, чтобы освободить место для миниатюр)
        plot_height = height * 0.4
        pdf.drawImage(temp_file, margin, height - margin - plot_height - 1 * cm,
                      width=width - 2 * margin, height=plot_height,
                      preserveAspectRatio=True, anchor='c')

        # Добавляем статистику
        pdf.setFont("Arial", 10)
        y_pos = height - margin - plot_height - 2 * cm

        pdf.drawString(margin, y_pos, f"Минимальный угол: {min_angle:.2f}°")
        pdf.drawString(width / 3, y_pos, f"Максимальный угол: {max_angle:.2f}°")
        pdf.drawString(2 * width / 3, y_pos, f"Средний угол: {mean_angle:.2f}°")

        # Добавляем миниатюры для минимального угла
        pdf.setFont("Arial-Bold", 10)
        pdf.drawString(margin, y_pos - 1.5 * cm, f"Пара с минимальным углом ({min_angle:.2f}°):")
        pdf.setFont("Arial", 8)

        if min_pair:
            thumb_size = 3 * cm
            img1_idx = min_pair['image1_idx'] - 1
            img2_idx = min_pair['image2_idx'] - 1

            # Первое изображение
            try:
                if 0 <= img1_idx < len(img_paths):
                    temp_file1 = os.path.join(temp_dir, f"min_thumb1.png")
                    with Image.open(img_paths[img1_idx]) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((int(thumb_size), int(thumb_size)))
                        img.save(temp_file1, 'PNG')
                    pdf.drawImage(temp_file1, margin, y_pos - 1.5 * cm - thumb_size - 0.2 * cm,
                                  width=thumb_size, height=thumb_size)
                    pdf.drawCentredString(margin + thumb_size / 2, y_pos - 1.5 * cm - thumb_size - 0.5 * cm,
                                          f"Изобр. {min_pair['image1_idx']}")
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения для min угла: {str(e)}")

            # Второе изображение
            try:
                if 0 <= img2_idx < len(img_paths):
                    temp_file2 = os.path.join(temp_dir, f"min_thumb2.png")
                    with Image.open(img_paths[img2_idx]) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((int(thumb_size), int(thumb_size)))
                        img.save(temp_file2, 'PNG')
                    pdf.drawImage(temp_file2, margin + thumb_size + 1 * cm, y_pos - 1.5 * cm - thumb_size - 0.2 * cm,
                                  width=thumb_size, height=thumb_size)
                    pdf.drawCentredString(margin + thumb_size + 1 * cm + thumb_size / 2,
                                          y_pos - 1.5 * cm - thumb_size - 0.5 * cm,
                                          f"Изобр. {min_pair['image2_idx']}")
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения для min угла: {str(e)}")

        # Добавляем миниатюры для максимального угла
        y_pos_min = y_pos - 1.5 * cm - thumb_size - 1 * cm
        pdf.setFont("Arial-Bold", 10)
        pdf.drawString(margin, y_pos_min, f"Пара с максимальным углом ({max_angle:.2f}°):")
        pdf.setFont("Arial", 8)

        if max_pair:
            thumb_size = 3 * cm
            img1_idx = max_pair['image1_idx'] - 1
            img2_idx = max_pair['image2_idx'] - 1

            # Первое изображение
            try:
                if 0 <= img1_idx < len(img_paths):
                    temp_file1 = os.path.join(temp_dir, f"max_thumb1.png")
                    with Image.open(img_paths[img1_idx]) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((int(thumb_size), int(thumb_size)))
                        img.save(temp_file1, 'PNG')
                    pdf.drawImage(temp_file1, margin, y_pos_min - thumb_size - 0.2 * cm,
                                  width=thumb_size, height=thumb_size)
                    pdf.drawCentredString(margin + thumb_size / 2, y_pos_min - thumb_size - 0.5 * cm,
                                          f"Изобр. {max_pair['image1_idx']}")
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения для max угла: {str(e)}")

            # Второе изображение
            try:
                if 0 <= img2_idx < len(img_paths):
                    temp_file2 = os.path.join(temp_dir, f"max_thumb2.png")
                    with Image.open(img_paths[img2_idx]) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((int(thumb_size), int(thumb_size)))
                        img.save(temp_file2, 'PNG')
                    pdf.drawImage(temp_file2, margin + thumb_size + 1 * cm, y_pos_min - thumb_size - 0.2 * cm,
                                  width=thumb_size, height=thumb_size)
                    pdf.drawCentredString(margin + thumb_size + 1 * cm + thumb_size / 2,
                                          y_pos_min - thumb_size - 0.5 * cm,
                                          f"Изобр. {max_pair['image2_idx']}")
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения для max угла: {str(e)}")

    except Exception as e:
        logger.error(f"Ошибка создания графика углов: {str(e)}")
        pdf.setFont("Arial", 10)
        pdf.drawString(margin, height - margin - 1 * cm, f"Ошибка создания графика: {str(e)}")
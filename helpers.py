# helpers.py


import datetime
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QPolygon

def parse_video_filename(filename: str) -> datetime.datetime:
    """
    Парсит имя видеофайла вида 'YYYYMMDDHHMMSSmmm' (или 'YYYYMMDDHHMMSS')
    и возвращает datetime.
    Пример: '20250221092234800' -> datetime(2025, 2, 21, 9, 22, 34, 800000)
    """
    # Если у файла есть расширение, его следует удалить перед вызовом этой функции.
    # Проверяем длину строки: 14 символов — без миллисекунд, 17 — с миллисекундами.
    if len(filename) == 14:
        return datetime.datetime.strptime(filename, "%Y%m%d%H%M%S")
    elif len(filename) == 17:
        return datetime.datetime.strptime(filename, "%Y%m%d%H%M%S%f")
    else:
        print("[WARNING] Имя файла с видео не соответствует формату. Возвращаем текущее время.")
        return datetime.datetime.now()

def format_datetime(dt: datetime.datetime) -> str:
    """
    Форматирует datetime обратно в строку вида 'YYYYMMDDHHMMSSmmm'.
    Пример: datetime(2025, 2, 21, 9, 22, 34, 800000) -> '20250221092234800'
    """
    ms = dt.microsecond // 1000  # миллисекунды
    # В оригинальном коде используется только strftime без миллисекунд,
    # но можно добавить миллисекунды, если потребуется.
    return dt.strftime("%Y%m%d%H%M%S")  # можно изменить на: return dt.strftime("%Y%m%d%H%M%S") + f"{ms:03d}"

def convert_roi_to_polygon(roi):
    """
    Преобразует ROI в QPolygon.
    Принимает:
      - QPolygon (возвращается без изменений)
      - Кортеж (x, y, w, h)
      - Объект с методом getRect() (например, QRect)
    """
    if roi is None:
        return None
    if isinstance(roi, QPolygon):
        return roi
    if isinstance(roi, tuple) and len(roi) == 4:
        x, y, w, h = roi
        return QPolygon([
            QPoint(x, y),
            QPoint(x + w, y),
            QPoint(x + w, y + h),
            QPoint(x, y + h)
        ])
    if hasattr(roi, 'getRect'):
        x, y, w, h = roi.getRect()
        return QPolygon([
            QPoint(x, y),
            QPoint(x + w, y),
            QPoint(x + w, y + h),
            QPoint(x, y + h)
        ])
    raise ValueError("Неизвестный тип ROI")

# config.py
import os 

# Базовый путь (папка, где находится config.py)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Сервер
SERVER_HOST = "192.168.67.189"
SERVER_PORT = 5000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# База данных
DATABASE_FILE = "cameras.db"

# YOLO модель
YOLO_MODEL_PATH = os.path.join(BASE_PATH, "models", "yolo11m.pt")

# Папка для снимков
SNAPSHOT_FOLDER = os.path.join(BASE_PATH, "snapshots")
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Папка для записи видео
VIDEO_RECORDING_FOLDER = os.path.join(BASE_PATH, "recordings")
os.makedirs(VIDEO_RECORDING_FOLDER, exist_ok=True)

# Путь к папке buttons, где лежат иконки
BUTTONS_PATH = os.path.join(BASE_PATH, "buttons")

# Формируем пути к иконкам для кнопок
START_ICON_PATH = os.path.join(BUTTONS_PATH, "player_play.svg")
STOP_ICON_PATH = os.path.join(BUTTONS_PATH, "pause_player.svg")

# Значения по умолчанию
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # 🔧 ИЗМЕНИТЬ ЗДЕСЬ: Порог уверенности для детекции людей (0.0-1.0)
DEFAULT_FRAME_INTERVAL = 4

# === НАСТРОЙКИ РАСПОЗНАВАНИЯ ===
# Основные флаги включения/выключения функций
ENABLE_FACE_RECOGNITION = True      # Распознавание лиц (MTCNN + InceptionResnetV1)
ENABLE_PERSON_RECOGNITION = True    # Распознавание людей (StrongSORT + ReID)
ENABLE_FACE_PERSON_LINKING = True   # Связывание лиц и людей
ENABLE_TRACKING_HISTORY = True      # Сохранение истории треков
ENABLE_PERSISTENT_STORAGE = True    # Сохранение эмбеддингов в БД

# Настройки распознавания лиц
FACE_RECOGNITION = True
FACE_THRESHOLD = 0.8  # 🔧 ИЗМЕНИТЬ ЗДЕСЬ: Порог уверенности для распознавания лиц (0.0-1.0)
MODEL_PATHS = {
    'mtcnn': None,  # Путь к кастомным весам MTCNN (если есть)
    'facenet': None  # Путь к кастомным весам FaceNet (если есть)
}

# === StrongSORT Tracking настройки ===
TRACKING_ENABLED = True
TRACKING_METHOD = "strongsort"  # strongsort, botsort, deepocsort, ocsort, bytetrack
REID_MODEL_PATH = "osnet_ibn_x1_0_msmt17.pt"  # Путь к ReID модели

# Параметры трекинга
TRACKING_CONFIG = {
    'max_age': 30,  # Максимальное количество кадров без детекции
    'n_init': 3,  # Количество кадров для подтверждения трека (уменьшили для более быстрого отклика)
    'max_cos_dist': 0.2,  # Максимальное косинусное расстояние для ReID
    'max_iou_dist': 0.7,  # Максимальное IoU расстояние для ассоциации
    'nn_budget': 100,  # Размер библиотеки признаков
    'mc_lambda': 0.98,  # Вес для согласованности движения
    'ema_alpha': 0.9,  # Альфа для экспоненциального скользящего среднего
}

# === Пороги для распознавания ===
PERSON_RECOGNITION_THRESHOLD = 0.5  # 🔧 ИЗМЕНИТЬ ЗДЕСЬ: Порог для распознавания людей (0.0-1.0)
FACE_PERSON_LINKING_THRESHOLD = 50  # Расстояние в пикселях для связи лица и человека
FACE_PERSON_LINKING_CONFIDENCE = 0.7  # 🔧 ИЗМЕНИТЬ ЗДЕСЬ: Уверенность в связи лица и человека (0.0-1.0)

# === НАСТРОЙКИ ЗАПИСИ ВИДЕО ===
VIDEO_RECORDING_ENABLED = True      # Включение/выключение записи видео
VIDEO_RECORDING_FPS = 30            # FPS для записи видео
VIDEO_RECORDING_CODEC = 'mp4v'      # Кодек для записи (mp4v, avc1, XVID)
VIDEO_RECORDING_QUALITY = 80        # Качество записи (0-100)
VIDEO_RECORDING_MAX_DURATION = 3600 # Максимальная длительность записи в секундах (1 час)

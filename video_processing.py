# video_processing.py

import cv2
import time
import numpy as np
import os
import base64
import traceback
import sqlite3
import torch
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QImage
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import DATABASE_FILE
from config import SNAPSHOT_FOLDER, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_FRAME_INTERVAL, YOLO_MODEL_PATH, FACE_THRESHOLD
from config import VIDEO_RECORDING_FOLDER, VIDEO_RECORDING_FPS, VIDEO_RECORDING_CODEC, VIDEO_RECORDING_QUALITY, VIDEO_RECORDING_MAX_DURATION
from network import send_screenshot_to_server, send_exceeded_stats, send_exceeded_stats_with_photo
from helpers import convert_roi_to_polygon, parse_video_filename, format_datetime
from database import (insert_error_log, save_periodic_stats, insert_face, find_closest_face, update_face_seen,
                      insert_person, find_closest_person, update_person_seen, link_face_to_person,
                      save_tracking_history, generate_session_id, get_statistics, get_person_by_face, get_face_name)
import logging
from database import log_camera_exceedance
from PIL import Image, ImageDraw, ImageFont

# === BoxMOT и StrongSORT импорты ===
import sys
sys.path.append('Yolov5_StrongSORT_OSNet')
from boxmot import create_tracker, get_tracker_config
from pathlib import Path

# Импортируем настройки
from config import (ENABLE_FACE_RECOGNITION, ENABLE_PERSON_RECOGNITION, 
                   ENABLE_FACE_PERSON_LINKING, ENABLE_TRACKING_HISTORY,
                   PERSON_RECOGNITION_THRESHOLD, FACE_PERSON_LINKING_THRESHOLD,
                   FACE_PERSON_LINKING_CONFIDENCE)

logger = logging.getLogger(__name__)


# Загружаем модель YOLO, используя путь из конфига.
model = YOLO(YOLO_MODEL_PATH)

# === Face detection & recognition ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Класс для записи видео ===
class VideoRecorder:
    """
    Класс для асинхронной записи видео с детекцией.
    """
    def __init__(self, output_path, fps=VIDEO_RECORDING_FPS, codec=VIDEO_RECORDING_CODEC, quality=VIDEO_RECORDING_QUALITY):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.writer = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        self.max_duration = VIDEO_RECORDING_MAX_DURATION
        
    def start_recording(self, frame_width, frame_height):
        """
        Начинает запись видео.
        
        Args:
            frame_width: Ширина кадра
            frame_height: Высота кадра
        """
        try:
            # Определяем кодек
            if self.codec == 'mp4v':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif self.codec == 'avc1':
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            elif self.codec == 'XVID':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Создаем VideoWriter
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (frame_width, frame_height)
            )
            
            if not self.writer.isOpened():
                raise Exception(f"Не удалось создать VideoWriter для {self.output_path}")
            
            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            
            logger.info(f"Запись видео начата: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при начале записи видео: {str(e)}")
            return False
    
    def write_frame(self, frame):
        """
        Записывает кадр в видео.
        
        Args:
            frame: Кадр для записи (numpy array)
        """
        if not self.is_recording or self.writer is None:
            return False
        
        try:
            # Проверяем максимальную длительность
            if self.start_time and (time.time() - self.start_time) > self.max_duration:
                logger.info("Достигнута максимальная длительность записи")
                self.stop_recording()
                return False
            
            self.writer.write(frame)
            self.frame_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при записи кадра: {str(e)}")
            return False
    
    def stop_recording(self):
        """
        Останавливает запись видео.
        """
        try:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            
            self.is_recording = False
            
            if self.start_time:
                duration = time.time() - self.start_time
                logger.info(f"Запись видео остановлена. Длительность: {duration:.2f}с, кадров: {self.frame_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при остановке записи видео: {str(e)}")
            return False
    
    def is_active(self):
        """
        Проверяет, активна ли запись.
        """
        return self.is_recording and self.writer is not None

# === StrongSORT Tracker инициализация ===
def initialize_tracker():
    """Инициализирует StrongSORT трекер с ReID моделью"""
    try:
        from config import TRACKING_ENABLED, TRACKING_METHOD, REID_MODEL_PATH, TRACKING_CONFIG
        
        if not TRACKING_ENABLED:
            logger.info("Tracking is disabled in config")
            return None
        
        # Путь к ReID модели
        reid_weights = Path(REID_MODEL_PATH)
        
        if not reid_weights.exists():
            logger.error(f"ReID model not found: {reid_weights}")
            return None
        
        # Получаем конфигурацию трекера
        tracker_config = get_tracker_config(TRACKING_METHOD)
        
        # Создаем трекер с пользовательскими параметрами
        tracker = create_tracker(
            tracker_type=TRACKING_METHOD,
            tracker_config=tracker_config,
            reid_weights=reid_weights,
            device=0,
            half=False,  # Используем полную точность
            per_class=False,  # Не разделяем по классам
            evolve_param_dict=TRACKING_CONFIG  # Передаем пользовательские параметры
        )
        
        logger.info(f"{TRACKING_METHOD.upper()} tracker initialized successfully")
        return tracker
    except Exception as e:
        logger.error(f"Failed to initialize {TRACKING_METHOD} tracker: {str(e)}")
        return None

# Инициализируем трекер
tracker = initialize_tracker()

# Глобальные переменные для отслеживания людей между сессиями
current_session_id = generate_session_id()
person_track_mapping = {}  # {track_id: person_id}
face_person_mapping = {}   # {face_id: person_id}

# --- Закомментирована функция определения позы ---
# def determine_person_position(x1, y1, x2, y2):
#     """
#     Определяет положение человека (стоит, сидит, лежит) на основе соотношения сторон его ограничивающего прямоугольника.
#     Args:
#         x1, y1: Координаты верхнего левого угла прямоугольника
#         x2, y2: Координаты нижнего правого угла прямоугольника
#     Returns:
#         str: 'standing' (стоит), 'sitting' (сидит) или 'lying' (лежит)
#     """
#     width = x2 - x1
#     height = y2 - y1
#     if width == 0 or height == 0:
#         return "unknown"
#     aspect_ratio = height / width
#     if aspect_ratio > 1.8:  # Высокий и узкий прямоугольник
#         return "standing"
#     elif aspect_ratio > 1.2:  # Среднее соотношение
#         return "sitting"
#     else:  # Широкий и низкий прямоугольник
#         return "lying"

def process_tracking(detections, frame):
    """
    Обрабатывает детекции через StrongSORT трекер.
    
    Args:
        detections: Результаты детекции YOLO
        frame: Текущий кадр
        
    Returns:
        tuple: (tracked_objects, people_count)
    """
    if tracker is None:
        # Если трекер не инициализирован, возвращаем обычную детекцию
        people_count = 0
        tracked_objects = []
        
        for det in detections[0].boxes.data:
            if len(det) < 6:
                continue  # пропускаем некорректную детекцию
            # Конвертируем тензор в CPU и затем в numpy
            det_cpu = det.cpu().numpy()
            x1, y1, x2, y2, score, cls = det_cpu
            if int(cls) == 0 and score >= DEFAULT_CONFIDENCE_THRESHOLD:
                tracked_objects.append([x1, y1, x2, y2, -1, score, cls])  # -1 означает отсутствие ID
                people_count += 1
        
        return tracked_objects, people_count
    
    try:
        # Подготавливаем детекции для трекера
        dets = []
        for det in detections[0].boxes.data:
            if len(det) < 6:
                continue  # пропускаем некорректную детекцию
            # Конвертируем тензор в CPU и затем в numpy
            det_cpu = det.cpu().numpy()
            x1, y1, x2, y2, score, cls = det_cpu
            if int(cls) == 0 and score >= DEFAULT_CONFIDENCE_THRESHOLD:
                dets.append([x1, y1, x2, y2, score, cls])
        
        if len(dets) == 0:
            # Обновляем трекер даже если нет детекций
            tracked_objects = tracker.update(np.empty((0, 6)), frame)
            return tracked_objects, 0
        
        # Конвертируем в numpy array
        dets = np.array(dets)
        
        # Обновляем трекер
        tracked_objects = tracker.update(dets, frame)
        
        # Подсчитываем количество людей
        people_count = len(tracked_objects) if len(tracked_objects) > 0 else 0
        
        return tracked_objects, people_count
        
    except Exception as e:
        logger.error(f"Error in tracking: {str(e)}")
        # Возвращаем обычную детекцию в случае ошибки
        people_count = 0
        tracked_objects = []
        
        for det in detections[0].boxes.data:
            if len(det) < 6:
                continue  # пропускаем некорректную детекцию
            x1, y1, x2, y2, score, cls = det
            if int(cls) == 0 and score >= DEFAULT_CONFIDENCE_THRESHOLD:
                tracked_objects.append([x1, y1, x2, y2, -1, score, cls])
                people_count += 1
        
        return tracked_objects, people_count

def process_person_recognition(tracked_objects, frame, camera_name=None, hospital_name=None):
    """
    Обрабатывает распознавание людей и связывание с лицами.
    
    Args:
        tracked_objects: Результаты трекинга
        frame: Текущий кадр
        camera_name: Имя камеры
        hospital_name: Имя больницы
        
    Returns:
        list: Обновленные объекты с информацией о распознавании
    """
    global current_session_id, person_track_mapping, face_person_mapping
    
    # Если распознавание людей отключено, возвращаем объекты без изменений
    if not ENABLE_PERSON_RECOGNITION:
        return tracked_objects
    
    if tracker is None:
        return tracked_objects
    
    try:
        updated_objects = []
        timestamp = datetime.utcnow().isoformat()
        
        for obj in tracked_objects:
            if len(obj) >= 7:  # Проверяем, что объект содержит все необходимые данные
                x1, y1, x2, y2, track_id, confidence, cls = obj[:7]
                
                # Извлекаем ReID эмбеддинг из трекера
                if hasattr(tracker, 'trackers') and track_id in tracker.trackers:
                    tracker_obj = tracker.trackers[track_id]
                    if hasattr(tracker_obj, 'smooth_feat') and tracker_obj.smooth_feat is not None:
                        reid_embedding = tracker_obj.smooth_feat.cpu().numpy().astype(np.float32)
                        reid_embedding_bytes = reid_embedding.tobytes()
                        
                        # Ищем существующего человека
                        existing_person_id = find_closest_person(reid_embedding_bytes, PERSON_RECOGNITION_THRESHOLD, current_session_id)
                        
                        if existing_person_id is not None:
                            # Обновляем время последнего появления
                            update_person_seen(existing_person_id, timestamp)
                            person_track_mapping[track_id] = existing_person_id
                            
                            # Сохраняем историю трека (если включено)
                            if ENABLE_TRACKING_HISTORY:
                                bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                                save_tracking_history(existing_person_id, track_id, current_session_id, 
                                                     bbox, confidence, camera_name, hospital_name)
                            
                            # Добавляем информацию о распознавании
                            obj = list(obj)
                            obj.append(existing_person_id)  # person_id
                            obj.append("recognized")  # status
                        else:
                            # Создаем нового человека
                            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                            new_person_id = insert_person(reid_embedding_bytes, timestamp, confidence,
                                                        bbox, track_id, current_session_id, 
                                                        camera_name, hospital_name)
                            

                            
                            if new_person_id:
                                person_track_mapping[track_id] = new_person_id
                                
                                # Сохраняем историю трека (если включено)
                                if ENABLE_TRACKING_HISTORY:
                                    save_tracking_history(new_person_id, track_id, current_session_id, 
                                                         bbox, confidence, camera_name, hospital_name)
                                
                                # Добавляем информацию о распознавании
                                obj = list(obj)
                                obj.append(new_person_id)  # person_id
                                obj.append("new")  # status
                            else:
                                obj = list(obj)
                                obj.append(None)  # person_id
                                obj.append("error")  # status
                    else:
                        obj = list(obj)
                        obj.append(None)  # person_id
                        obj.append("no_embedding")  # status
                else:
                    obj = list(obj)
                    obj.append(None)  # person_id
                    obj.append("no_tracker")  # status
                
                updated_objects.append(obj)
            else:
                updated_objects.append(obj)
        
        return updated_objects
        
    except Exception as e:
        logger.error(f"Error in person recognition: {str(e)}")
        return tracked_objects

def process_face_recognition_and_linking(frame, tracked_objects, camera_name=None, hospital_name=None):
    """
    Обрабатывает распознавание лиц и связывание с людьми.
    
    Args:
        frame: Текущий кадр
        tracked_objects: Результаты трекинга людей
        camera_name: Имя камеры
        hospital_name: Имя больницы
        
    Returns:
        tuple: (face_objects, linked_count)
    """
    global face_person_mapping
    
    # Если распознавание лиц отключено, возвращаем пустые результаты
    if not ENABLE_FACE_RECOGNITION:
        return [], 0
    
    try:
        # Конвертируем кадр для MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Детекция лиц
        boxes, _ = mtcnn.detect(frame_rgb)
        
        face_objects = []
        linked_count = 0
        timestamp = datetime.utcnow().isoformat()
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Извлекаем лицо
                face = frame_rgb[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                
                # Получаем эмбеддинг лица
                from PIL import Image
                from torchvision import transforms
                face_img = Image.fromarray(face)
                preprocess = transforms.Compose([
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                face_tensor = preprocess(face_img).to(device)
                face_tensor = face_tensor.unsqueeze(0)
                embedding = resnet(face_tensor).detach().cpu().numpy()
                embedding_bytes = embedding.astype(np.float32).tobytes()
                
                # Ищем существующее лицо
                existing_face_id = find_closest_face(embedding_bytes, FACE_THRESHOLD)
                
                if existing_face_id is not None:
                    # Обновляем время последнего появления
                    update_face_seen(existing_face_id, timestamp)
                    
                    # Проверяем, связано ли лицо с человеком
                    linked_person_id = get_person_by_face(existing_face_id)
                    
                    if linked_person_id:
                        face_objects.append({
                            'bbox': [x1, y1, x2, y2],
                            'face_id': existing_face_id,
                            'person_id': linked_person_id,
                            'status': 'linked'
                        })
                        linked_count += 1
                    else:
                        face_objects.append({
                            'bbox': [x1, y1, x2, y2],
                            'face_id': existing_face_id,
                            'person_id': None,
                            'status': 'recognized'
                        })
                else:
                    # Создаем новое лицо
                    face_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    new_face_id = insert_face(embedding_bytes, timestamp, 0.8, face_bbox)
                    

                    
                    if new_face_id:
                        face_objects.append({
                            'bbox': [x1, y1, x2, y2],
                            'face_id': new_face_id,
                            'person_id': None,
                            'status': 'new'
                        })
                        
                        # Пытаемся связать с ближайшим человеком (если включено)
                        if ENABLE_FACE_PERSON_LINKING:
                            closest_person = find_closest_person_in_frame(x1, y1, x2, y2, tracked_objects)
                            if closest_person:
                                link_face_to_person(new_face_id, closest_person['person_id'], FACE_PERSON_LINKING_CONFIDENCE)
                                face_objects[-1]['person_id'] = closest_person['person_id']
                                face_objects[-1]['status'] = 'linked'
                                linked_count += 1
                                
                                # Связываем лицо с ID трекинга
                                if closest_person['track_id'] is not None:
                                    link_face_to_tracking_id(new_face_id, closest_person['track_id'], FACE_PERSON_LINKING_CONFIDENCE)

        return face_objects, linked_count
        
    except Exception as e:
        logger.error(f"Error in face recognition: {str(e)}")
        return [], 0

def find_closest_person_in_frame(face_x1, face_y1, face_x2, face_y2, tracked_objects, threshold=None):
    if threshold is None:
        threshold = FACE_PERSON_LINKING_THRESHOLD
    """
    Находит ближайшего человека в кадре к лицу.
    
    Args:
        face_x1, face_y1, face_x2, face_y2: Координаты лица
        tracked_objects: Результаты трекинга людей
        threshold: Порог расстояния для связи
        
    Returns:
        dict: Информация о ближайшем человеке или None
    """
    try:
        closest_person = None
        min_distance = float('inf')
        
        for obj in tracked_objects:
            if len(obj) >= 7:
                person_x1, person_y1, person_x2, person_y2, track_id, confidence, cls = obj[:7]
                
                # Вычисляем расстояние между центром лица и центром человека
                face_center_x = (face_x1 + face_x2) / 2
                face_center_y = (face_y1 + face_y2) / 2
                person_center_x = (person_x1 + person_x2) / 2
                person_center_y = (person_y1 + person_y2) / 2
                
                distance = ((face_center_x - person_center_x) ** 2 + 
                           (face_center_y - person_center_y) ** 2) ** 0.5
                
                if distance < min_distance and distance <= threshold:
                    min_distance = distance
                    person_id = obj[7] if len(obj) > 7 else None
                    closest_person = {
                        'person_id': person_id,
                        'track_id': track_id,
                        'distance': distance
                    }
        
        return closest_person
        
    except Exception as e:
        logger.error(f"Error finding closest person: {str(e)}")
        return None

def async_save_send_and_record(hospital_name, camera_name, snapshot, people_count, max_people_count):
    """
    Асинхронно сохраняет скриншот, записывает данные в БД и отправляет на сервер и в Telegram.
    """
    try:
        # Создаем папку для снимков
        os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
        
        # Генерируем имя файла с временной меткой
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        snapshot_filename = f"{SNAPSHOT_FOLDER}/camera_{camera_name}_{timestamp}.jpg"
        
        # Сохраняем скриншот
        if not cv2.imwrite(snapshot_filename, snapshot, [cv2.IMWRITE_JPEG_QUALITY, 60]):
            raise IOError(f"Failed to save snapshot: {snapshot_filename}")
        
        # Кодируем фото в base64 для отправки в Telegram
        ret_enc, buffer = cv2.imencode(".jpg", snapshot, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ret_enc:
            raise IOError("Failed to encode snapshot")
        photo_data = base64.b64encode(buffer).decode("utf-8")
        
        # Подготавливаем данные для отправки
        stats_data = {
            "camera_name": camera_name,
            "hospital_name": hospital_name,
            "people_count": people_count,
            "max_people_count": max_people_count,
            "timestamp": timestamp,
            "photo_data": photo_data
        }
        
        # Отправляем данные на сервер
        server_success = send_exceeded_stats_with_photo(
            hospital_name,
            camera_name,
            people_count,
            max_people_count,
            photo_data
        )
        
        # Отправляем скриншот на сервер
        screenshot_success = send_screenshot_to_server(
            hospital_name,
            camera_name,
            snapshot_filename
        )
        
        # Записываем в БД
        db_success = log_camera_exceedance(
            camera_name,
            hospital_name,
            people_count,
            max_people_count,
            timestamp
        )
        
        # Проверяем успешность всех операций
        if not all([server_success, screenshot_success, db_success]):
            logger.error("Some operations failed during async_save_send_and_record")
            return False
            
        logger.info(f"Successfully processed exceedance event for camera {camera_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error in async_save_send_and_record: {str(e)}")
        return False

class ROISelector(QThread):
    """
    Поток для выбора ROI с использованием диалога OpenCV.
    Вызывает cv2.selectROI и передаёт выбранный ROI (в виде кортежа (x, y, w, h)) через сигнал roi_selected.
    """
    roi_selected = pyqtSignal(tuple)

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.frame = frame.copy()

    def run(self):
        roi = cv2.selectROI("Выберите область детекции", self.frame, False, False)
        cv2.destroyWindow("Выберите область детекции")
        self.roi_selected.emit(roi)

class VideoProcessingThread(QThread):
    """
    Поток для обработки видеофайла.
    Выполняет детекцию объектов на кадрах видео с использованием модели YOLO.
    Отправляет обработанный кадр через сигнал frame_processed и количество обнаруженных объектов через detection_signal.
    """
    frame_processed = pyqtSignal(QImage)
    detection_signal = pyqtSignal(int)
    error_occurred = pyqtSignal(str, int, str)  # (сообщение об ошибке, ID видео, stack trace)
    recording_status_changed = pyqtSignal(bool)  # Сигнал изменения статуса записи

    def __init__(self, video_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, roi=None, frame_interval=DEFAULT_FRAME_INTERVAL, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.confidence_threshold = confidence_threshold
        self.roi = roi  # Ожидается QPolygon, если выбрана произвольная область
        self.frame_interval = frame_interval
        self.running = True
        self.paused = False
        self.last_frame = None
        self.cached_roi_polygon = None
        self.cached_pts = None
        # Добавляем атрибуты для работы с видео
        self.video_id = id(self)  # Уникальный идентификатор для видео
        self.video_name = os.path.basename(video_path)
        self.hospital_name = "Видео"  # По умолчанию
        self.people_count = 0
        self.max_people_count = 0  # Будет установлено из UI
        self.exceed_start_time = None
        self.exceed_alert_sent = False
        # Таймер для периодической статистики
        self.timer = QTimer()
        self.timer.timeout.connect(self.log_periodic_stats)
        self.timer.start(30000)  # Каждые 30 секунд
        # Атрибуты для записи видео
        self.video_recorder = None
        self.is_recording = False
        # model и tracker уже инициализированы глобально

    def update_settings(self):
        """Обновляет настройки в реальном времени"""
        # Переимпортируем конфигурацию для получения актуальных значений
        import config
        global ENABLE_FACE_RECOGNITION, ENABLE_PERSON_RECOGNITION, ENABLE_FACE_PERSON_LINKING
        global ENABLE_TRACKING_HISTORY, ENABLE_PERSISTENT_STORAGE
        
        ENABLE_FACE_RECOGNITION = config.ENABLE_FACE_RECOGNITION
        ENABLE_PERSON_RECOGNITION = config.ENABLE_PERSON_RECOGNITION
        ENABLE_FACE_PERSON_LINKING = config.ENABLE_FACE_PERSON_LINKING
        ENABLE_TRACKING_HISTORY = config.ENABLE_TRACKING_HISTORY
        ENABLE_PERSISTENT_STORAGE = config.ENABLE_PERSISTENT_STORAGE
        
        print(f"[VideoProcessingThread] Настройки обновлены: Face={ENABLE_FACE_RECOGNITION}, Person={ENABLE_PERSON_RECOGNITION}, Linking={ENABLE_FACE_PERSON_LINKING}")

    def get_db_connection(self):
        """Создает новое соединение с базой данных в текущем потоке"""
        return sqlite3.connect(DATABASE_FILE)

    def update_cached_roi(self):
        # Вызывайте этот метод, когда ROI изменилось (например, после выбора в диалоге)
        from helpers import convert_roi_to_polygon
        try:
            self.cached_roi_polygon = convert_roi_to_polygon(self.roi)
            pts = []
            for i in range(self.cached_roi_polygon.count()):
                pt = self.cached_roi_polygon.point(i)
                pts.append([pt.x(), pt.y()])
            self.cached_pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        except Exception as e:
            print("Ошибка преобразования ROI:", e)
            self.cached_roi_polygon = None
            self.cached_pts = None

    def log_periodic_stats(self):
        try:
            from datetime import datetime
            stats_data = {
                "camera_name": self.video_name,
                "hospital_name": self.hospital_name,
                "people_count": self.people_count,
                "timestamp": datetime.now().isoformat()
            }
            # Сохранение статистики в базу данных
            save_periodic_stats(self.video_name, self.hospital_name, self.people_count)
            # Отправка статистики на сервер
            from network import NetworkWorker, send_stats_to_server
            if not hasattr(self, "network_workers"):
                self.network_workers = []
            worker = NetworkWorker(send_stats_to_server, stats_data)
            worker.finished_signal.connect(lambda result, w=worker: self.network_workers.remove(w))
            self.network_workers.append(worker)
            worker.start()
            print("log_periodic_stats: статистика отправлена")
        except Exception as e:
            print(f"Ошибка в log_periodic_stats: {e}")

    def log_error(self, error_message, stack_trace):
        try:
            insert_error_log(error_message, self.video_id, self.video_name, stack_trace)
        except Exception as e:
            print(f"Ошибка при записи в логи: {e}")
        self.error_occurred.emit(error_message, self.video_id, stack_trace)

    def log_exceeded_people(self, people_count):
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            max_people = self.max_people_count
            video_name = self.video_name
            hospital_name = self.hospital_name
            cursor.execute("""
                INSERT INTO people_exceeded 
                (camera_name, hospital_name, timestamp, people_count, max_people_count)
                VALUES (?, ?, datetime('now'), ?, ?)
            """, (video_name, hospital_name, people_count, max_people))
            conn.commit()
            conn.close()
            logger.info("Запись о превышении успешно сохранена в БД")
            
            # Получаем путь к последнему сохранённому снимку для данной больницы
            screenshot_path = self.get_latest_screenshot(hospital_name)
            photo_data = None
            if screenshot_path and os.path.exists(screenshot_path):
                with open(screenshot_path, "rb") as f:
                    import base64
                    photo_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Отправляем данные через NetworkWorker с бинарными данными
            if not hasattr(self, 'network_workers'):
                self.network_workers = []
            from network import NetworkWorker, send_exceeded_stats_with_photo
            worker = NetworkWorker(send_exceeded_stats_with_photo, hospital_name, video_name, people_count, max_people, photo_data)
            worker.finished_signal.connect(lambda result, w=worker: self.network_workers.remove(w))
            self.network_workers.append(worker)
            worker.start()

        except sqlite3.Error as e:
            logger.error(f"Ошибка при записи в таблицу people_exceeded: {e}")

    def get_latest_screenshot(self, hospital_name):
        # Получаем путь к последнему сохраненному снимку для данной больницы
        try:
            snapshot_dir = SNAPSHOT_FOLDER
            if not os.path.exists(snapshot_dir):
                return None
                
            files = [os.path.join(snapshot_dir, f) for f in os.listdir(snapshot_dir) 
                    if f.endswith('.jpg') and hospital_name in f]
            
            if not files:
                return None
                
            # Возвращаем самый новый файл
            return max(files, key=os.path.getctime)
        except Exception as e:
            print(f"Ошибка при получении последнего снимка: {e}")
            return None

    def save_snapshot(self, frame):
        if frame is None:
            return
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{SNAPSHOT_FOLDER}/video_{self.video_id}_{timestamp}.jpg"
            resized_frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
            max_text = f"MAX.PEOPLE: {self.max_people_count}"
            now_text = f"NOW PEOPLE: {self.people_count}"
            color_max = (255, 255, 255)
            color_now = (0, 0, 255)
            cv2.putText(resized_frame, max_text, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_max, 1, cv2.LINE_AA)
            (text_width, _) = cv2.getTextSize(max_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            offset = text_width + 60
            cv2.putText(resized_frame, now_text, (10 + offset, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_now, 1, cv2.LINE_AA)
            cv2.imwrite(filename, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            print(f"Сохранен снимок: {filename}")
            success = send_screenshot_to_server(self.hospital_name, self.video_name, filename)
            if success:
                print(f"Снимок отправлен: {filename}")
            else:
                print(f"Ошибка отправки снимка: {filename}")
        except Exception as e:
            self.log_error(f"Ошибка при сохранении снимка: {str(e)}", traceback.format_exc())

    def run(self):
        import cv2, time, traceback, base64, io, numpy as np, concurrent.futures
        cap = cv2.VideoCapture(self.video_path)
        try:
            if not cap.isOpened():
                self.log_error("Не удалось открыть видео: " + self.video_path, traceback.format_exc())
                return
            else:
                print(f"[DEBUG] Видео открыто успешно: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps > 0 else 30
            frame_interval_sec = 1.0 / fps
            next_frame_time = time.time()
            frame_count = 0
            exceed_threshold = 5  # Порог времени для длительного превышения (в секундах)

            # Если ROI уже задан, обновляем кэш
            if self.roi is not None:
                self.update_cached_roi()

            # Создаем пул потоков для инференса, чтобы не блокировать основной цикл
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                while self.running and cap.isOpened():
                    if self.paused:
                        self.msleep(100)
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        print("[DEBUG] Видео закончилось или возникла ошибка при чтении кадра.")
                        break

                    self.last_frame = frame.copy()
                    # Увеличиваем счетчик кадров
                    frame_count += 1

                    if frame_count % self.frame_interval == 0:
                        inference_start = time.time()
                        
                        # Выполняем детекцию только если распознавание людей включено
                        if ENABLE_PERSON_RECOGNITION:
                            # Запускаем инференс в пуле потоков
                            future = executor.submit(model, frame, classes=[0], verbose=False)
                            try:
                                results = future.result(timeout=10)
                            except Exception as e:
                                self.log_error("Ошибка инференса", str(e))
                                continue
                            # --- ФИЛЬТРАЦИЯ bounding boxes по ROI до трекера ---
                            if self.roi is not None:
                                try:
                                    from helpers import convert_roi_to_polygon
                                    import numpy as np
                                    import cv2
                                    import torch
                                    roi_polygon = convert_roi_to_polygon(self.roi)
                                    pts = []
                                    for i in range(roi_polygon.count()):
                                        pt = roi_polygon.point(i)
                                        pts.append([pt.x(), pt.y()])
                                    pts_np = np.array(pts, np.int32)
                                    def in_poly(x1, y1, x2, y2):
                                        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                                        return cv2.pointPolygonTest(pts_np, (cx, cy), False) >= 0
                                    filtered_boxes = []
                                    for det in results[0].boxes.data:
                                        if len(det) < 6:
                                            continue
                                        det_cpu = det.cpu().numpy()
                                        x1, y1, x2, y2, score, cls = det_cpu
                                        if int(cls) == 0 and score >= DEFAULT_CONFIDENCE_THRESHOLD and in_poly(x1, y1, x2, y2):
                                            filtered_boxes.append(det)
                                    if filtered_boxes:
                                        results[0].boxes.data = torch.stack(filtered_boxes)
                                    else:
                                        results[0].boxes.data = torch.empty((0,6))
                                except Exception as e:
                                    self.log_error("Ошибка фильтрации по ROI до трекера", str(e))
                            # --- конец фильтрации по ROI до трекера ---
                            inference_time = time.time() - inference_start
                            print(f"[DEBUG] Inference time: {inference_time:.2f} sec")

                            # Используем StrongSORT трекинг
                            tracked_objects, people_count = process_tracking(results, frame)
                        else:
                            # Если распознавание людей отключено, не выполняем детекцию
                            tracked_objects = []
                            people_count = 0
                            print("[DEBUG] Распознавание людей отключено, детекция пропущена")
                        
                        # Обрабатываем распознавание людей (если включено)
                        if ENABLE_PERSON_RECOGNITION:
                            tracked_objects = process_person_recognition(tracked_objects, frame, 
                                                                       self.video_name, self.hospital_name)
                        
                            # Обрабатываем распознавание лиц и связывание (если включено)
                            if ENABLE_FACE_RECOGNITION:
                                face_objects, linked_count = process_face_recognition_and_linking(
                                    frame, tracked_objects, self.video_name, self.hospital_name)
                                # --- ФИЛЬТРАЦИЯ лиц по ROI ---
                                if self.roi is not None:
                                    try:
                                        from helpers import convert_roi_to_polygon
                                        import numpy as np
                                        import cv2
                                        roi_polygon = convert_roi_to_polygon(self.roi)
                                        pts = []
                                        for i in range(roi_polygon.count()):
                                            pt = roi_polygon.point(i)
                                            pts.append([pt.x(), pt.y()])
                                        pts_np = np.array(pts, np.int32)
                                        def in_poly(x1, y1, x2, y2):
                                            cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                                            return cv2.pointPolygonTest(pts_np, (cx, cy), False) >= 0
                                        face_objects = [face_obj for face_obj in face_objects if in_poly(face_obj['bbox'][0], face_obj['bbox'][1], face_obj['bbox'][2], face_obj['bbox'][3])]
                                    except Exception as e:
                                        self.log_error("Ошибка фильтрации лиц по ROI", str(e))
                                # --- конец фильтрации лиц по ROI ---
                            else:
                                face_objects, linked_count = [], 0
                            
                            # --- ВРЕМЕННАЯ СВЯЗЬ TRACK_ID <-> FACE_ID ПО КООРДИНАТАМ ---
                            track_to_face = {}
                            if ENABLE_FACE_RECOGNITION:
                                for face_obj in face_objects:
                                    fx1, fy1, fx2, fy2 = face_obj['bbox']
                                    face_center = ((fx1 + fx2) // 2, (fy1 + fy2) // 2)
                                    min_dist = float('inf')
                                    best_track = None
                                    for obj in tracked_objects:
                                        if len(obj) >= 7:
                                            x1, y1, x2, y2, track_id, score, cls = obj[:7]
                                            track_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                            dist = ((face_center[0] - track_center[0]) ** 2 + (face_center[1] - track_center[1]) ** 2) ** 0.5
                                            if dist < min_dist and dist < 50:
                                                min_dist = dist
                                                best_track = track_id
                                    if best_track is not None:
                                        track_to_face[best_track] = face_obj['face_id']

                            # --- ОТРИСОВКА КВАДРАТА И ID ДЛЯ КАЖДОГО ЧЕЛОВЕКА ---
                            for obj in tracked_objects:
                                if len(obj) >= 7:
                                    x1, y1, x2, y2, track_id, score, cls = obj[:7]
                                    face_id = track_to_face.get(track_id)
                                    if face_id:
                                        from database import get_face_name
                                        face_name = get_face_name(face_id)
                                        if face_name:
                                            label = f"Человек ID: {track_id} | Лицо: {face_id} ({face_name})"
                                        else:
                                            label = f"Человек ID: {track_id} | Лицо: {face_id}"
                                    else:
                                        label = f"Человек ID: {track_id}" if track_id is not None and track_id >= 0 else "Человек ID: ?"
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                    frame = draw_text_with_unicode(frame, label, (int(x1), int(y1) - 25), (0, 255, 0), 24)

                            # Визуализация лиц (только если распознавание лиц включено)
                            if ENABLE_FACE_RECOGNITION:
                                for face_obj in face_objects:
                                    fx1, fy1, fx2, fy2 = face_obj['bbox']
                                    face_id = face_obj['face_id']
                                    person_id = face_obj['person_id']
                                    status = face_obj['status']
                                    
                                    # Получаем имя лица из базы данных
                                    from database import get_face_name
                                    face_name = get_face_name(face_id)
                                    
                                    # Выбираем цвет для лица
                                    if status == "linked":
                                        face_color = (255, 0, 255)  # Пурпурный - связано с человеком
                                        if face_name:
                                            face_text = f"{face_id} ({face_name})"
                                        else:
                                            face_text = f"{face_id}"
                                    elif status == "recognized":
                                        face_color = (0, 255, 255)  # Желтый - распознанное лицо
                                        if face_name:
                                            face_text = f"{face_id} ({face_name})"
                                        else:
                                            face_text = f"{face_id}"
                                    else:  # new
                                        face_color = (255, 255, 0)  # Голубой - новое лицо
                                        if face_name:
                                            face_text = f"{face_id} ({face_name})"
                                        else:
                                            face_text = f"{face_id}"
                                    
                                    # Рисуем рамку лица
                                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
                                    frame = draw_text_with_unicode(frame, face_text, (fx1, fy1 - 30), face_color, 24)

                        self.people_count = people_count
                        self.last_frame = frame.copy()

                        # Логируем текущие значения
                        print(f"[DEBUG] Detected people: {people_count}, Max allowed: {self.max_people_count}")

                        # Если количество людей превышает максимум, выполняем запись и отправку уведомления
                        if self.max_people_count > 0 and people_count > self.max_people_count:
                            if self.exceed_start_time is None:
                                self.exceed_start_time = time.time()
                            elif time.time() - self.exceed_start_time >= exceed_threshold and not self.exceed_alert_sent:
                                # Сохраняем событие превышения в базу данных
                                from database import log_video_event
                                event_time_str = time.strftime("%Y%m%d-%H%M%S")
                                log_video_event(self.video_name, people_count, self.max_people_count, event_time_str)
                                print(f"[DEBUG] Exceeded event logged at {event_time_str}")
                                
                                # Кодирование последнего кадра в JPEG и преобразование в base64
                                ret_enc, buffer = cv2.imencode(".jpg", self.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                                if ret_enc:
                                    photo_data = base64.b64encode(buffer).decode("utf-8")
                                else:
                                    photo_data = None
                                
                                # Отправляем уведомление через NetworkWorker с бинарными данными
                                from network import NetworkWorker, send_exceeded_stats_with_photo
                                worker = NetworkWorker(send_exceeded_stats_with_photo,
                                                    self.hospital_name,
                                                    self.video_name,
                                                    people_count,
                                                    self.max_people_count,
                                                    photo_data)
                                if not hasattr(self, 'network_workers'):
                                    self.network_workers = []
                                self.network_workers.append(worker)
                                worker.finished_signal.connect(lambda result, w=worker: self.network_workers.remove(w))
                                worker.start()
                                self.exceed_alert_sent = True
                                
                                # Сохраняем скриншот
                                self.save_snapshot(self.last_frame)
                        else:
                            self.exceed_start_time = None
                            self.exceed_alert_sent = False

                        # Масштабирование кадра для отображения
                        frame_h, frame_w = frame.shape[:2]
                        disp_w = self.stream_screen.width() if hasattr(self, 'stream_screen') else frame_w
                        disp_h = self.stream_screen.height() if hasattr(self, 'stream_screen') else frame_h
                        scaled_frame = cv2.resize(frame, (disp_w, disp_h))
                        
                        if self.roi is not None:
                            try:
                                from helpers import convert_roi_to_polygon
                                roi_polygon = convert_roi_to_polygon(self.roi)
                                pts = []
                                for i in range(roi_polygon.count()):
                                    pt = roi_polygon.point(i)
                                    scaled_x = int(pt.x() * (disp_w / frame_w))
                                    scaled_y = int(pt.y() * (disp_h / frame_h))
                                    pts.append([scaled_x, scaled_y])
                                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                                cv2.polylines(scaled_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                            except Exception as e:
                                self.log_error("Ошибка отображения ROI", str(e))
                        
                        # Записываем кадр в видео, если запись активна
                        if self.is_recording and self.video_recorder is not None:
                            self.video_recorder.write_frame(frame)
                        
                        # Отправляем сигналы с обработанным кадром и количеством людей
                        frame_rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
                        h_img, w_img, ch = frame_rgb.shape
                        bytes_per_line = ch * w_img
                        q_img = QImage(frame_rgb.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888)
                        self.frame_processed.emit(q_img)
                        self.detection_signal.emit(people_count)

                    next_frame_time += frame_interval_sec
                    current_time = time.time()
                    delay = next_frame_time - current_time
                    if delay > 0:
                        self.msleep(int(delay * 1000))
                    else:
                        next_frame_time = current_time

                cap.release()

        except Exception as e:
            self.log_error(f"Ошибка в видео {self.video_name}: {str(e)}", traceback.format_exc())
        finally:
            if cap is not None and cap.isOpened():
                cap.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def start_video_recording(self):
        """
        Начинает запись видео с детекцией.
        """
        try:
            if self.is_recording:
                logger.warning("Запись видео уже активна")
                return False
            
            if self.last_frame is None:
                logger.error("Нет кадра для записи")
                return False
            
            # Создаем имя файла для записи
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = f"video_recording_{self.video_name}_{timestamp}.mp4"
            output_path = os.path.join(VIDEO_RECORDING_FOLDER, output_filename)
            
            # Создаем VideoRecorder
            self.video_recorder = VideoRecorder(output_path)
            
            # Получаем размеры кадра
            frame_height, frame_width = self.last_frame.shape[:2]
            
            # Начинаем запись
            if self.video_recorder.start_recording(frame_width, frame_height):
                self.is_recording = True
                self.recording_status_changed.emit(True)
                logger.info(f"Запись видео начата: {output_path}")
                return True
            else:
                self.video_recorder = None
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при начале записи видео: {str(e)}")
            return False
    
    def stop_video_recording(self):
        """
        Останавливает запись видео.
        """
        try:
            if not self.is_recording or self.video_recorder is None:
                return False
            
            success = self.video_recorder.stop_recording()
            if success:
                self.is_recording = False
                self.recording_status_changed.emit(False)
                logger.info("Запись видео остановлена")
            
            self.video_recorder = None
            return success
            
        except Exception as e:
            logger.error(f"Ошибка при остановке записи видео: {str(e)}")
            return False
    
    def is_video_recording(self):
        """
        Проверяет, активна ли запись видео.
        """
        return self.is_recording and self.video_recorder is not None

    def stop(self):
        # Останавливаем запись видео перед завершением потока
        if self.is_recording:
            self.stop_video_recording()
        
        self.running = False
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
        self.wait()

class CameraThread(QThread):
    """
    Поток для обработки видеопотока с камеры.
    Выполняет детекцию объектов, логирование статистики и ошибок, а также сохранение и отправку снимков при превышении лимита.
    """
    frame_signal = pyqtSignal(int, int)  # (количество людей, ID камеры)
    frame_display_signal = pyqtSignal(int, QImage)  # (ID камеры, кадр для отображения)
    error_occurred = pyqtSignal(str, int, str)  # (сообщение об ошибке, ID камеры, stack trace)
    recording_status_changed = pyqtSignal(int, bool)  # (ID камеры, статус записи)

    def __init__(self, camera_id, camera_data, frame_interval=DEFAULT_FRAME_INTERVAL, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        super().__init__()
        self.camera_id = camera_id
        self.camera_data = camera_data
        self.frame_interval = frame_interval
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.max_people_count = camera_data.get('max_people_count', 0)
        self.hospital_name = camera_data.get('hospital_name', 'Не указано')
        self.roi = None
        self.cached_roi_polygon = None
        self.people_count = 0
        self.last_frame = None
        self.exceed_start_time = None
        self.exceed_alert_sent = False
        self.last_periodic_log_time = 0
        # Таймер для периодической статистики
        self.timer = QTimer()
        self.timer.timeout.connect(self.log_periodic_stats)
        self.timer.start(30000)  # Обновление каждые 30 секунд
        # Переменные для управления паузой
        self.cap = None
        # Атрибуты для записи видео
        self.video_recorder = None
        self.is_recording = False
        # model и tracker уже инициализированы глобально

    def update_settings(self):
        """Обновляет настройки в реальном времени"""
        # Переимпортируем конфигурацию для получения актуальных значений
        import config
        global ENABLE_FACE_RECOGNITION, ENABLE_PERSON_RECOGNITION, ENABLE_FACE_PERSON_LINKING
        global ENABLE_TRACKING_HISTORY, ENABLE_PERSISTENT_STORAGE
        
        ENABLE_FACE_RECOGNITION = config.ENABLE_FACE_RECOGNITION
        ENABLE_PERSON_RECOGNITION = config.ENABLE_PERSON_RECOGNITION
        ENABLE_FACE_PERSON_LINKING = config.ENABLE_FACE_PERSON_LINKING
        ENABLE_TRACKING_HISTORY = config.ENABLE_TRACKING_HISTORY
        ENABLE_PERSISTENT_STORAGE = config.ENABLE_PERSISTENT_STORAGE
        
        print(f"[CameraThread] Настройки обновлены: Face={ENABLE_FACE_RECOGNITION}, Person={ENABLE_PERSON_RECOGNITION}, Linking={ENABLE_FACE_PERSON_LINKING}")

    def get_db_connection(self):
        """Создает новое соединение с базой данных в текущем потоке"""
        return sqlite3.connect(DATABASE_FILE)

    def update_cached_roi(self):
        """
        Обновление кэша выбранного ROI. Преобразует ROI в QPolygon и сохраняет координаты точек.
        """
        try:
            self.cached_roi_polygon = convert_roi_to_polygon(self.roi)
            pts = []
            for i in range(self.cached_roi_polygon.count()):
                pt = self.cached_roi_polygon.point(i)
                pts.append([pt.x(), pt.y()])
            self.cached_pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        except Exception as e:
            print("Ошибка преобразования ROI:", e)
            self.cached_roi_polygon = None
            self.cached_pts = None

    def log_periodic_stats(self):
        try:
            from datetime import datetime
            stats_data = {
                "camera_name": self.camera_data['name'],
                "hospital_name": self.camera_data.get('hospital_name', 'Не указано'),
                "people_count": self.people_count,
                "timestamp": datetime.now().isoformat()
            }
            # Сохранение статистики в базу данных
            save_periodic_stats(self.camera_data['name'], self.camera_data.get('hospital_name', 'Не указано'), self.people_count)
            # Отправка статистики на сервер
            from network import NetworkWorker, send_stats_to_server
            if not hasattr(self, "network_workers"):
                self.network_workers = []
            worker = NetworkWorker(send_stats_to_server, stats_data)
            worker.finished_signal.connect(lambda result, w=worker: self.network_workers.remove(w))
            self.network_workers.append(worker)
            worker.start()
            print("log_periodic_stats: статистика отправлена")
        except Exception as e:
            print(f"Ошибка в log_periodic_stats: {e}")



    def run(self):
        try:
            # Формирование URL подключения к камере
            if self.camera_data.get("path"):
                url = (f"{self.camera_data['protocol']}://{self.camera_data['username']}:"
                       f"{self.camera_data['password']}@{self.camera_data['ip']}:"
                       f"{self.camera_data['port']}{self.camera_data['path']}")
            else:
                url = (f"{self.camera_data['protocol']}://{self.camera_data['username']}:"
                       f"{self.camera_data['password']}@{self.camera_data['ip']}:"
                       f"{self.camera_data['port']}/")
            
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                self.log_error("Не удалось подключиться к камере", traceback.format_exc())
                return

            self.running = True
            frame_count = 0
            exceed_threshold = 5  # Порог времени для длительного превышения (в секундах)
            self.exceed_start_time = None
            self.exceed_alert_sent = False

            # Если ROI задано, обновляем кэш
            if self.roi is not None:
                self.update_cached_roi()

            while self.running:
                if not self.cap.isOpened():
                    self.log_error("Соединение с камерой потеряно", traceback.format_exc())
                    break

                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame_count += 1
                if frame_count % self.frame_interval == 0:
                    # Обработка ROI, если оно задано и включено распознавание людей
                    if self.roi is not None and ENABLE_PERSON_RECOGNITION:
                        try:
                            from helpers import convert_roi_to_polygon
                            roi_polygon = convert_roi_to_polygon(self.roi)
                        except Exception as e:
                            self.log_error("Ошибка преобразования ROI", str(e))
                            results = model(frame, classes=[0], verbose=False)
                        else:
                            rect = roi_polygon.boundingRect()
                            roi_x, roi_y, roi_w, roi_h = rect.x(), rect.y(), rect.width(), rect.height()
                            frame_h, frame_w = frame.shape[:2]
                            if roi_x < 0 or roi_y < 0 or roi_x + roi_w > frame_w or roi_y + roi_h > frame_h:
                                self.log_error("Некорректная область (ROI)", "ROI выходит за пределы кадра")
                                results = model(frame, classes=[0], verbose=False)
                            else:
                                frame_roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                                try:
                                    results = model(frame_roi, classes=[0], verbose=False)
                                    for i, box in enumerate(results[0].boxes.data):
                                        x1, y1, x2, y2, score, cls = box.tolist()
                                        # Корректировка координат для полного кадра
                                        results[0].boxes.data[i][0] = x1 + roi_x
                                        results[0].boxes.data[i][1] = y1 + roi_y
                                        results[0].boxes.data[i][2] = x2 + roi_x
                                        results[0].boxes.data[i][3] = y2 + roi_y
                                except Exception as e:
                                    self.log_error("Ошибка детекции на ROI", str(e))
                                    results = model(frame, classes=[0], verbose=False)
                    elif ENABLE_PERSON_RECOGNITION:
                        # Если распознавание людей включено, выполняем детекцию
                        results = model(frame, classes=[0], verbose=False)
                        tracked_objects, people_count = process_tracking(results, frame)
                        tracked_objects = process_person_recognition(tracked_objects, frame, 
                                                                   self.camera_data['name'], self.hospital_name)
                        self.people_count = people_count
                        face_objects = []
                        linked_count = 0
                        # Только лица, если включено
                        if ENABLE_FACE_RECOGNITION:
                            face_objects, linked_count = process_face_recognition_and_linking(
                                frame, tracked_objects, self.camera_data['name'], self.hospital_name)
                            # --- ФИЛЬТРАЦИЯ лиц по ROI ---
                            if self.roi is not None:
                                try:
                                    from helpers import convert_roi_to_polygon
                                    import numpy as np
                                    import cv2
                                    roi_polygon = convert_roi_to_polygon(self.roi)
                                    pts = []
                                    for i in range(roi_polygon.count()):
                                        pt = roi_polygon.point(i)
                                        pts.append([pt.x(), pt.y()])
                                    pts_np = np.array(pts, np.int32)
                                    def in_poly(x1, y1, x2, y2):
                                        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                                        return cv2.pointPolygonTest(pts_np, (cx, cy), False) >= 0
                                    face_objects = [face_obj for face_obj in face_objects if in_poly(face_obj['bbox'][0], face_obj['bbox'][1], face_obj['bbox'][2], face_obj['bbox'][3])]
                                except Exception as e:
                                    self.log_error("Ошибка фильтрации лиц по ROI", str(e))
                            # --- конец фильтрации лиц по ROI ---
                        else:
                            face_objects, linked_count = [], 0
                        # --- ВРЕМЕННАЯ СВЯЗЬ TRACK_ID <-> FACE_ID ПО КООРДИНАТАМ ---
                        track_to_face = {}
                        if ENABLE_FACE_RECOGNITION:
                            for face_obj in face_objects:
                                fx1, fy1, fx2, fy2 = face_obj['bbox']
                                face_center = ((fx1 + fx2) // 2, (fy1 + fy2) // 2)
                                min_dist = float('inf')
                                best_track = None
                                for obj in tracked_objects:
                                    if len(obj) >= 7:
                                        x1, y1, x2, y2, track_id, score, cls = obj[:7]
                                        track_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                        dist = ((face_center[0] - track_center[0]) ** 2 + (face_center[1] - track_center[1]) ** 2) ** 0.5
                                        if dist < min_dist and dist < 50:
                                            min_dist = dist
                                            best_track = track_id
                                if best_track is not None:
                                    track_to_face[best_track] = face_obj['face_id']
                        # --- ОТРИСОВКА КВАДРАТА И ID ДЛЯ КАЖДОГО ЧЕЛОВЕКА ---
                        for obj in tracked_objects:
                            if len(obj) >= 7:
                                x1, y1, x2, y2, track_id, score, cls = obj[:7]
                                face_id = track_to_face.get(track_id)
                                if face_id:
                                    from database import get_face_name
                                    face_name = get_face_name(face_id)
                                    if face_name:
                                        label = f"Человек ID: {track_id} | Лицо: {face_id} ({face_name})"
                                    else:
                                        label = f"Человек ID: {track_id} | Лицо: {face_id}"
                                else:
                                    label = f"Человек ID: {track_id}" if track_id is not None and track_id >= 0 else "Человек ID: ?"
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                frame = draw_text_with_unicode(frame, label, (int(x1), int(y1) - 25), (0, 255, 0), 24)
                        
                        # Визуализация лиц (если включено распознавание лиц)
                        if ENABLE_FACE_RECOGNITION:
                            for face_obj in face_objects:
                                fx1, fy1, fx2, fy2 = face_obj['bbox']
                                face_id = face_obj['face_id']
                                person_id = face_obj['person_id']
                                status = face_obj['status']
                                
                                # Получаем имя лица из базы данных
                                from database import get_face_name
                                face_name = get_face_name(face_id)
                                
                                if status == "linked":
                                    face_color = (255, 0, 255)  # Пурпурный - связано с человеком
                                    if face_name:
                                        face_text = f"{face_id} ({face_name})"
                                    else:
                                        face_text = f"{face_id}"
                                elif status == "recognized":
                                    face_color = (0, 255, 255)  # Желтый - распознанное лицо
                                    if face_name:
                                        face_text = f"{face_id} ({face_name})"
                                    else:
                                        face_text = f"{face_id}"
                                else:  # new
                                    face_color = (255, 255, 0)  # Голубой - новое лицо
                                    if face_name:
                                        face_text = f"{face_id} ({face_name})"
                                    else:
                                        face_text = f"{face_id}"
                                
                                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
                                frame = draw_text_with_unicode(frame, face_text, (fx1, fy1 - 30), face_color, 24)
                        self.last_frame = frame.copy()
                        # --- Передача кадра в GUI ---
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h_img, w_img, ch = frame_rgb.shape
                        bytes_per_line = ch * w_img
                        q_img = QImage(frame_rgb.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888)
                        self.frame_display_signal.emit(self.camera_id, q_img)
                        continue
                    else:
                        # Если распознавание людей отключено, не делаем детекцию и не рисуем рамки
                        tracked_objects = []
                        people_count = 0
                        face_objects = []
                        linked_count = 0
                        # Только лица, если включено
                        if ENABLE_FACE_RECOGNITION:
                            face_objects, linked_count = process_face_recognition_and_linking(
                                frame, tracked_objects, self.camera_data['name'], self.hospital_name)
                            # --- ФИЛЬТРАЦИЯ лиц по ROI ---
                            if self.roi is not None:
                                try:
                                    from helpers import convert_roi_to_polygon
                                    import numpy as np
                                    import cv2
                                    roi_polygon = convert_roi_to_polygon(self.roi)
                                    pts = []
                                    for i in range(roi_polygon.count()):
                                        pt = roi_polygon.point(i)
                                        pts.append([pt.x(), pt.y()])
                                    pts_np = np.array(pts, np.int32)
                                    def in_poly(x1, y1, x2, y2):
                                        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                                        return cv2.pointPolygonTest(pts_np, (cx, cy), False) >= 0
                                    face_objects = [face_obj for face_obj in face_objects if in_poly(face_obj['bbox'][0], face_obj['bbox'][1], face_obj['bbox'][2], face_obj['bbox'][3])]
                                except Exception as e:
                                    self.log_error("Ошибка фильтрации лиц по ROI", str(e))
                            # --- конец фильтрации лиц по ROI ---
                        else:
                            face_objects, linked_count = [], 0
                        # Визуализация лиц (если включено распознавание лиц)
                        if ENABLE_FACE_RECOGNITION:
                            for face_obj in face_objects:
                                fx1, fy1, fx2, fy2 = face_obj['bbox']
                                face_id = face_obj['face_id']
                                person_id = face_obj['person_id']
                                status = face_obj['status']
                                
                                # Получаем имя лица из базы данных
                                from database import get_face_name
                                face_name = get_face_name(face_id)
                                
                                if status == "linked":
                                    face_color = (255, 0, 255)  # Пурпурный - связано с человеком
                                    if face_name:
                                        face_text = f"{face_id} ({face_name})"
                                    else:
                                        face_text = f"{face_id}"
                                elif status == "recognized":
                                    face_color = (0, 255, 255)  # Желтый - распознанное лицо
                                    if face_name:
                                        face_text = f"{face_id} ({face_name})"
                                    else:
                                        face_text = f"{face_id}"
                                else:  # new
                                    face_color = (255, 255, 0)  # Голубой - новое лицо
                                    if face_name:
                                        face_text = f"{face_id} ({face_name})"
                                    else:
                                        face_text = f"{face_id}"
                                
                                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
                                frame = draw_text_with_unicode(frame, face_text, (fx1, fy1 - 30), face_color, 24)
                        self.people_count = 0
                        self.last_frame = frame.copy()
                        # --- Передача кадра в GUI ---
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h_img, w_img, ch = frame_rgb.shape
                        bytes_per_line = ch * w_img
                        q_img = QImage(frame_rgb.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888)
                        self.frame_display_signal.emit(self.camera_id, q_img)
                        continue

                    self.people_count = people_count
                    self.last_frame = frame.copy()

                    # Логика сохранения скриншота при превышении количества людей
                    if people_count > self.max_people_count:
                        if self.exceed_start_time is None:
                            self.exceed_start_time = time.time()
                        elif time.time() - self.exceed_start_time >= exceed_threshold and not self.exceed_alert_sent:
                            snapshot = self.last_frame.copy()
                            from network import NetworkWorker
                            worker = NetworkWorker(
                                async_save_send_and_record,
                                self.hospital_name,
                                self.camera_data['name'],
                                snapshot,
                                people_count,
                                self.max_people_count
                            )
                            if not hasattr(self, 'network_workers'):
                                self.network_workers = []
                            self.network_workers.append(worker)
                            worker.finished_signal.connect(lambda result, w=worker: self.network_workers.remove(w))
                            worker.start()
                            print(f"[DEBUG] Событие превышения для камеры обработано асинхронно в {time.strftime('%Y%m%d-%H%M%S')}")
                            self.exceed_alert_sent = True
                    else:
                        self.exceed_start_time = None
                        self.exceed_alert_sent = False

                    # Масштабирование кадра для отображения
                    frame_h, frame_w = frame.shape[:2]
                    disp_w = self.stream_screen.width() if hasattr(self, 'stream_screen') else frame_w
                    disp_h = self.stream_screen.height() if hasattr(self, 'stream_screen') else frame_h
                    scaled_frame = cv2.resize(frame, (disp_w, disp_h))
                    if self.roi is not None:
                        try:
                            from helpers import convert_roi_to_polygon
                            roi_polygon = convert_roi_to_polygon(self.roi)
                            pts = []
                            for i in range(roi_polygon.count()):
                                pt = roi_polygon.point(i)
                                scaled_x = int(pt.x() * (disp_w / frame_w))
                                scaled_y = int(pt.y() * (disp_h / frame_h))
                                pts.append([scaled_x, scaled_y])
                            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(scaled_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                        except Exception as e:
                            self.log_error("Ошибка отображения ROI", str(e))

                    # Записываем кадр в видео, если запись активна
                    if self.is_recording and self.video_recorder is not None:
                        self.video_recorder.write_frame(frame)
                    
                    self.frame_signal.emit(people_count, self.camera_id)
                    frame_rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
                    h_img, w_img, ch = frame_rgb.shape
                    bytes_per_line = ch * w_img
                    q_img = QImage(frame_rgb.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888)
                    self.frame_display_signal.emit(self.camera_id, q_img)

                    # --- ФИЛЬТРАЦИЯ ПО ROI (ПОЛИГОНУ) ---
                    if self.roi is not None:
                        try:
                            from helpers import convert_roi_to_polygon
                            import numpy as np
                            import cv2
                            roi_polygon = convert_roi_to_polygon(self.roi)
                            pts = []
                            for i in range(roi_polygon.count()):
                                pt = roi_polygon.point(i)
                                pts.append([pt.x(), pt.y()])
                            pts_np = np.array(pts, np.int32)
                            def in_poly(x1, y1, x2, y2):
                                cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                                return cv2.pointPolygonTest(pts_np, (cx, cy), False) >= 0
                            # Фильтруем людей
                            tracked_objects = [obj for obj in tracked_objects if len(obj) >= 4 and in_poly(obj[0], obj[1], obj[2], obj[3])]
                            # people_count только по отфильтрованным
                            people_count = len(tracked_objects)
                            # Фильтруем лица, если есть
                            if ENABLE_FACE_RECOGNITION and 'face_objects' in locals():
                                face_objects = [face_obj for face_obj in face_objects if in_poly(face_obj['bbox'][0], face_obj['bbox'][1], face_obj['bbox'][2], face_obj['bbox'][3])]
                        except Exception as e:
                            self.log_error("Ошибка фильтрации по ROI", str(e))
        except Exception as e:
            logging.error(f"Ошибка камеры {self.camera_id}: {str(e)}\n{traceback.format_exc()}")
        finally:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None

    def start_video_recording(self):
        """
        Начинает запись видео с детекцией для камеры.
        """
        try:
            if self.is_recording:
                logger.warning(f"Запись видео уже активна для камеры {self.camera_id}")
                return False
            
            if self.last_frame is None:
                logger.error(f"Нет кадра для записи камеры {self.camera_id}")
                return False
            
            # Создаем имя файла для записи
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            camera_name = self.camera_data.get('name', f'camera_{self.camera_id}')
            output_filename = f"camera_recording_{camera_name}_{timestamp}.mp4"
            output_path = os.path.join(VIDEO_RECORDING_FOLDER, output_filename)
            
            # Создаем VideoRecorder
            self.video_recorder = VideoRecorder(output_path)
            
            # Получаем размеры кадра
            frame_height, frame_width = self.last_frame.shape[:2]
            
            # Начинаем запись
            if self.video_recorder.start_recording(frame_width, frame_height):
                self.is_recording = True
                self.recording_status_changed.emit(self.camera_id, True)
                logger.info(f"Запись видео камеры {self.camera_id} начата: {output_path}")
                return True
            else:
                self.video_recorder = None
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при начале записи видео камеры {self.camera_id}: {str(e)}")
            return False
    
    def stop_video_recording(self):
        """
        Останавливает запись видео для камеры.
        """
        try:
            if not self.is_recording or self.video_recorder is None:
                return False
            
            success = self.video_recorder.stop_recording()
            if success:
                self.is_recording = False
                self.recording_status_changed.emit(self.camera_id, False)
                logger.info(f"Запись видео камеры {self.camera_id} остановлена")
            
            self.video_recorder = None
            return success
            
        except Exception as e:
            logger.error(f"Ошибка при остановке записи видео камеры {self.camera_id}: {str(e)}")
            return False
    
    def is_video_recording(self):
        """
        Проверяет, активна ли запись видео для камеры.
        """
        return self.is_recording and self.video_recorder is not None

    def stop(self):
        # Останавливаем запись видео перед завершением потока
        if self.is_recording:
            self.stop_video_recording()
        
        self.running = False
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        # Ждем завершения потока
        self.wait(3000)

# --- FreeType для кириллицы ---
try:
    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName='C:/Windows/Fonts/arial.ttf', id=0)
except Exception as e:
    ft = None
    print('FreeType не инициализирован, подписи будут стандартные:', e)

# Универсальная функция для вывода текста с поддержкой кириллицы
def draw_text_with_unicode(frame, text, org, color=(0,255,255), font_height=24):
    # Сначала пробуем FreeType
    if ft:
        try:
            ft.putText(frame, text, org, fontHeight=font_height, color=color, thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=False)
            return frame
        except Exception as e:
            print('FreeType не смог отрисовать текст:', e)
    # Если не получилось — используем PIL
    try:
        # OpenCV: BGR, PIL: RGB
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Путь к шрифту с кириллицей
        font_path = 'C:/Windows/Fonts/arial.ttf'
        font = ImageFont.truetype(font_path, font_height)
        # PIL: цвет в RGB
        color_rgb = (color[2], color[1], color[0])
        draw.text(org, text, font=font, fill=color_rgb)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print('PIL не смог отрисовать текст:', e)
    return frame

def link_face_to_tracking_id(face_id, track_id, confidence=0.8):
    """
    Связывает ID лица с ID трекинга тела.
    
    Args:
        face_id: ID распознанного лица
        track_id: ID трекинга от StrongSORT
        confidence: Уверенность в связи
        
    Returns:
        bool: True если связь успешно создана
    """
    try:
        from database import execute_db_query
        
        # Сначала находим person_id по track_id из текущей сессии
        global person_track_mapping
        person_id = person_track_mapping.get(track_id)
        
        if person_id:
            # Связываем лицо с человеком через существующую таблицу
            query = """
                INSERT OR REPLACE INTO face_person_links (face_id, person_id, confidence, created_at)
                VALUES (?, ?, ?, datetime('now'))
            """
            success = execute_db_query(query, (face_id, person_id, confidence))
            
            if success:
                print(f"[DEBUG] Связано лицо {face_id} с треком {track_id} (person_id: {person_id})")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Ошибка связи лица с треком: {str(e)}")
        return False

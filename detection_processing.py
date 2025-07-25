# detection_processing.py

import cv2
import time
import numpy as np
import os
import base64
import traceback
import sqlite3
import torch
from datetime import datetime
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

def process_tracking(detections, frame):
    """
    Обрабатывает детекции через StrongSORT трекер.
    
    Args:
        detections: Результаты детекции YOLO
        frame: Текущий кадр
        
    Returns:
        tuple: (tracked_objects, people_count)
    """
    # Проверяем, есть ли детекции
    if detections is None or len(detections) == 0 or detections[0].boxes is None:
        pass  # Нет детекций для обработки
        return [], 0
    
    # Если трекер не инициализирован, возвращаем простую детекцию без трекинга
    if tracker is None:
        pass  # Трекер не инициализирован, используем простую детекцию
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
        
        pass  # Простая детекция завершена
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
        
        pass  # Детекции подготовлены для трекера
        
        if len(dets) == 0:
            # Обновляем трекер даже если нет детекций
            tracked_objects = tracker.update(np.empty((0, 6)), frame)
            pass  # Нет детекций выше порога
            return tracked_objects, 0
        
        # Конвертируем в numpy array
        dets = np.array(dets)
        
        # Обновляем трекер
        tracked_objects = tracker.update(dets, frame)
        
        # Подсчитываем количество людей
        people_count = len(tracked_objects) if len(tracked_objects) > 0 else 0
        
        pass  # Трекинг завершён
        return tracked_objects, people_count
        
    except Exception as e:
        logger.error(f"Ошибка в трекинге: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Возвращаем простую детекцию в случае ошибки трекера
        people_count = 0
        tracked_objects = []
        
        pass  # Трекер упал, используем простую детекцию
        for det in detections[0].boxes.data:
            if len(det) < 6:
                continue  # пропускаем некорректную детекцию
            det_cpu = det.cpu().numpy()
            x1, y1, x2, y2, score, cls = det_cpu
            if int(cls) == 0 and score >= DEFAULT_CONFIDENCE_THRESHOLD:
                tracked_objects.append([x1, y1, x2, y2, -1, score, cls])
                people_count += 1
        
        pass  # Простая детекция после ошибки завершена
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
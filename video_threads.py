# video_threads.py

import cv2
import time
import numpy as np
import os
import base64
import traceback
import sqlite3
import torch
import concurrent.futures
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QImage

from config import DATABASE_FILE
from config import SNAPSHOT_FOLDER, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_FRAME_INTERVAL, YOLO_MODEL_PATH, FACE_THRESHOLD
from config import VIDEO_RECORDING_FOLDER, VIDEO_RECORDING_FPS, VIDEO_RECORDING_CODEC, VIDEO_RECORDING_QUALITY, VIDEO_RECORDING_MAX_DURATION
from config import (ENABLE_FACE_RECOGNITION, ENABLE_PERSON_RECOGNITION, 
                   ENABLE_FACE_PERSON_LINKING, ENABLE_TRACKING_HISTORY,
                   PERSON_RECOGNITION_THRESHOLD, FACE_PERSON_LINKING_THRESHOLD,
                   FACE_PERSON_LINKING_CONFIDENCE)

from network import send_screenshot_to_server, NetworkWorker, send_stats_to_server, send_exceeded_stats_with_photo
from helpers import convert_roi_to_polygon, parse_video_filename, format_datetime
from database import (insert_error_log, save_periodic_stats, get_face_name, log_video_event)
import logging

# Импортируем функции обработки из detection_processing
from detection_processing import (
    model, tracker, mtcnn, resnet, VideoRecorder,
    process_tracking, process_person_recognition, process_face_recognition_and_linking,
    async_save_send_and_record, draw_text_with_unicode
)

logger = logging.getLogger(__name__)

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
            insert_error_log(error_message, self.camera_id, f"Camera_{self.camera_id}", stack_trace)
        except Exception as e:
            print(f"Ошибка при записи в логи: {e}")
        self.error_occurred.emit(error_message, self.camera_id, stack_trace)

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
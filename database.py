# database.py

import sqlite3
import logging
import numpy as np
import json
import uuid
from datetime import datetime
from config import (DATABASE_FILE, ENABLE_FACE_RECOGNITION, ENABLE_PERSON_RECOGNITION, 
                   ENABLE_FACE_PERSON_LINKING, ENABLE_TRACKING_HISTORY, ENABLE_PERSISTENT_STORAGE)
from network import NetworkWorker


def execute_db_query(query, params=()):
    """
    Выполняет запрос к базе данных с обработкой ошибок.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        logging.error(f"Ошибка выполнения запроса: {e}. Запрос: {query} | Параметры: {params}")
        return False

def log_camera_exceedance(camera_name, hospital_name, people_count, max_people_count, event_time_str):
    """
    Сохраняет событие превышения для трансляции с камеры в таблицу people_exceeded.
    """
    query = """
        INSERT INTO people_exceeded (camera_name, hospital_name, timestamp, people_count, max_people_count)
        VALUES (?, ?, ?, ?, ?)
    """
    return execute_db_query(query, (camera_name, hospital_name, event_time_str, people_count, max_people_count))



def log_video_event(video_name, people_count, max_people_count, event_time_str):
    """
    Записывает событие видео (например, превышение лимита людей) в таблицу video_exceedance_stats.
    """
    query = """
        INSERT INTO video_exceedance_stats (video_name, event_time, people_count, max_people_count)
        VALUES (?, ?, ?, ?)
    """
    return execute_db_query(query, (video_name, event_time_str, people_count, max_people_count))

def load_cameras():
    """
    Загружает все записи о камерах из таблицы cameras и возвращает список словарей.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cameras")
        cameras = []
        for row in cursor.fetchall():
            camera_data = {
                "id": row[0],
                "name": row[1],
                "ip": row[2],
                "port": row[3],
                "username": row[4],
                "password": row[5],
                "protocol": row[6],
                "path": row[7],
                "max_people_count": row[8]
            }
            cameras.append(camera_data)
        conn.close()
        return cameras
    except sqlite3.Error as e:
        print(f"Ошибка при загрузке камер из базы данных: {e}")
        return []

def save_cameras(cameras):
    """
    Сохраняет список камер в базу данных. При этом проверяется, что камера с таким IP ещё не добавлена.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        for camera in cameras:
            # Проверяем, существует ли камера с таким же IP
            cursor.execute("SELECT id FROM cameras WHERE ip = ?", (camera['ip'],))
            existing_camera = cursor.fetchone()
            if not existing_camera:
                cursor.execute("""
                    INSERT INTO cameras (name, ip, port, username, password, protocol, path, max_people_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    camera['name'], camera['ip'], camera['port'], camera['username'],
                    camera['password'], camera['protocol'], camera['path'],
                    camera['max_people_count']
                ))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Ошибка при сохранении камер в базу данных: {e}")

def insert_error_log(error_message, camera_id, camera_name, stack_trace):
    """
    Добавляет запись об ошибке в таблицу error_logs.
    """
    query = """
        INSERT INTO error_logs (error_message, camera_id, camera_name, stack_trace)
        VALUES (?, ?, ?, ?)
    """
    return execute_db_query(query, (error_message, camera_id, camera_name, stack_trace))

def save_periodic_stats(camera_name, hospital_name, people_count):
    """
    Сохраняет периодическую статистику по количеству людей для камеры.
    """
    query = """
        INSERT INTO periodic_stats (camera_name, hospital_name, people_count)
        VALUES (?, ?, ?)
    """
    return execute_db_query(query, (camera_name, hospital_name, people_count))

def load_error_logs():
    """
    Загружает все записи из таблицы error_logs, сортируя по убыванию времени.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM error_logs ORDER BY timestamp DESC")
        logs = cursor.fetchall()
        conn.close()
        return logs
    except sqlite3.Error as e:
        print(f"Ошибка при загрузке логов: {e}")
        return []

def clear_error_logs():
    """
    Удаляет все записи из таблицы error_logs.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM error_logs")
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Ошибка при очистке логов: {e}")
        return False

def register_user(chat_id, hospital_name):
    """
    Регистрирует пользователя Telegram с указанной больницей.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO users (chat_id, hospital_name)
            VALUES (?, ?)
        """, (chat_id, hospital_name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Ошибка при регистрации пользователя: {e}")
        return False

def get_users_by_hospital(hospital_name):
    """
    Получает список chat_id всех пользователей, зарегистрированных для указанной больницы.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT chat_id FROM users
            WHERE hospital_name = ?
        """, (hospital_name,))
        chat_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return chat_ids
    except sqlite3.Error as e:
        print(f"Ошибка при получении пользователей больницы: {e}")
        return []

def unregister_user(chat_id, hospital_name):
    """
    Удаляет регистрацию пользователя для указанной больницы.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM users
            WHERE chat_id = ? AND hospital_name = ?
        """, (chat_id, hospital_name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Ошибка при удалении регистрации пользователя: {e}")
        return False

def get_user_hospitals(chat_id):
    """
    Получает список больниц, на которые подписан пользователь.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT hospital_name FROM users
            WHERE chat_id = ?
        """, (chat_id,))
        hospitals = [row[0] for row in cursor.fetchall()]
        conn.close()
        return hospitals
    except sqlite3.Error as e:
        print(f"Ошибка при получении списка больниц пользователя: {e}")
        return []

# === Условные функции для лиц ===
def insert_face(embedding_bytes: bytes, timestamp: str, confidence: float = 0.0, 
                face_bbox: dict = None, person_id: int = None) -> int:
    """
    Вставляет новое лицо в базу данных и возвращает его ID.
    Работает только если ENABLE_FACE_RECOGNITION = True
    """
    if not ENABLE_FACE_RECOGNITION or not ENABLE_PERSISTENT_STORAGE:
        return None
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        bbox_json = json.dumps(face_bbox) if face_bbox else None
        
        cursor.execute("""
            INSERT INTO faces (embedding, first_seen, last_seen, confidence, face_bbox, person_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (embedding_bytes, timestamp, timestamp, confidence, bbox_json, person_id))
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return face_id
    except sqlite3.Error as e:
        logging.error(f"Ошибка при добавлении лица: {e}")
        return None

def find_closest_face(embedding_bytes: bytes, threshold: float) -> int:
    """
    Находит ближайшее лицо в базе данных по эмбеддингу.
    Работает только если ENABLE_FACE_RECOGNITION = True
    """
    if not ENABLE_FACE_RECOGNITION or not ENABLE_PERSISTENT_STORAGE:
        return None
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT face_id, embedding FROM faces")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        query_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        min_distance = float('inf')
        closest_face_id = None

        for face_id, stored_embedding_bytes in rows:
            stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)
            # Нормализуем векторы для косинусного расстояния
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
            # Вычисляем косинусное расстояние
            distance = 1 - np.dot(query_embedding_norm, stored_embedding_norm)
            
            if distance < min_distance:
                min_distance = distance
                closest_face_id = face_id

        return closest_face_id if min_distance <= threshold else None
    except Exception as e:
        logging.error(f"Ошибка при поиске ближайшего лица: {e}")
        return None

def update_face_seen(face_id: int, timestamp: str) -> None:
    """
    Обновляет время последнего появления лица.
    Работает только если ENABLE_FACE_RECOGNITION = True
    """
    if not ENABLE_FACE_RECOGNITION or not ENABLE_PERSISTENT_STORAGE:
        return
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE faces
            SET last_seen = ?
            WHERE face_id = ?
        """, (timestamp, face_id))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logging.error(f"Ошибка при обновлении времени появления лица: {e}")

def get_faces_count() -> int:
    """
    Возвращает количество уникальных лиц в базе данных.
    Работает только если ENABLE_FACE_RECOGNITION = True
    """
    if not ENABLE_FACE_RECOGNITION:
        return 0
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faces")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении количества лиц: {e}")
        return 0

# === Условные функции для людей ===
def insert_person(reid_embedding_bytes: bytes, timestamp: str, confidence: float = 0.0,
                  person_bbox: dict = None, track_id: int = None, session_id: str = None,
                  camera_name: str = None, hospital_name: str = None, name: str = None) -> int:
    """
    Вставляет нового человека в базу данных и возвращает его ID.
    Работает только если ENABLE_PERSON_RECOGNITION = True
    """
    if not ENABLE_PERSON_RECOGNITION or not ENABLE_PERSISTENT_STORAGE:
        return None
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        bbox_json = json.dumps(person_bbox) if person_bbox else None
        
        cursor.execute("""
            INSERT INTO persons (reid_embedding, first_seen, last_seen, confidence, 
                                person_bbox, track_id, session_id, camera_name, hospital_name, name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (reid_embedding_bytes, timestamp, timestamp, confidence, bbox_json, 
              track_id, session_id, camera_name, hospital_name, name))
        person_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return person_id
    except sqlite3.Error as e:
        logging.error(f"Ошибка при добавлении человека: {e}")
        return None

def find_closest_person(reid_embedding_bytes: bytes, threshold: float, 
                       session_id: str = None) -> int:
    """
    Находит ближайшего человека в базе данных по ReID эмбеддингу.
    Работает только если ENABLE_PERSON_RECOGNITION = True
    """
    if not ENABLE_PERSON_RECOGNITION or not ENABLE_PERSISTENT_STORAGE:
        return None
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("SELECT person_id, reid_embedding FROM persons WHERE session_id != ?", (session_id,))
        else:
            cursor.execute("SELECT person_id, reid_embedding FROM persons")
            
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        query_embedding = np.frombuffer(reid_embedding_bytes, dtype=np.float32)
        min_distance = float('inf')
        closest_person_id = None

        for person_id, stored_embedding_bytes in rows:
            stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)
            # Нормализуем векторы для косинусного расстояния
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
            # Вычисляем косинусное расстояние
            distance = 1 - np.dot(query_embedding_norm, stored_embedding_norm)
            
            if distance < min_distance:
                min_distance = distance
                closest_person_id = person_id

        return closest_person_id if min_distance <= threshold else None
    except Exception as e:
        logging.error(f"Ошибка при поиске ближайшего человека: {e}")
        return None

def update_person_seen(person_id: int, timestamp: str) -> None:
    """
    Обновляет время последнего появления человека.
    Работает только если ENABLE_PERSON_RECOGNITION = True
    """
    if not ENABLE_PERSON_RECOGNITION or not ENABLE_PERSISTENT_STORAGE:
        return
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE persons
            SET last_seen = ?
            WHERE person_id = ?
        """, (timestamp, person_id))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logging.error(f"Ошибка при обновлении времени появления человека: {e}")

def get_persons_count() -> int:
    """
    Возвращает количество уникальных людей в базе данных.
    Работает только если ENABLE_PERSON_RECOGNITION = True
    """
    if not ENABLE_PERSON_RECOGNITION:
        return 0
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM persons")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении количества людей: {e}")
        return 0

# === Условные функции для связывания ===
def link_face_to_person(face_id: int, person_id: int, confidence: float = 0.0) -> bool:
    """
    Связывает лицо с человеком.
    Работает только если ENABLE_FACE_PERSON_LINKING = True
    """
    if not ENABLE_FACE_PERSON_LINKING:
        return False
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO face_person_links (face_id, person_id, confidence, linked_at)
            VALUES (?, ?, ?, ?)
        """, (face_id, person_id, confidence, now))
        
        # Обновляем person_id в таблице faces
        cursor.execute("""
            UPDATE faces SET person_id = ? WHERE face_id = ?
        """, (person_id, face_id))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        logging.error(f"Ошибка при связывании лица с человеком: {e}")
        return False

def get_person_by_face(face_id: int) -> int:
    """
    Получает person_id по face_id.
    Работает только если ENABLE_FACE_PERSON_LINKING = True
    """
    if not ENABLE_FACE_PERSON_LINKING:
        return None
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT person_id FROM faces WHERE face_id = ?", (face_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении человека по лицу: {e}")
        return None

def get_faces_by_person(person_id: int) -> list:
    """
    Получает список face_id для данного person_id.
    Работает только если ENABLE_FACE_PERSON_LINKING = True
    """
    if not ENABLE_FACE_PERSON_LINKING:
        return []
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT face_id FROM faces WHERE person_id = ?", (person_id,))
        face_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return face_ids
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении лиц по человеку: {e}")
        return []

# === Условные функции для истории треков ===
def save_tracking_history(person_id: int, track_id: int, session_id: str, 
                         bbox: dict, confidence: float = 0.0, 
                         camera_name: str = None, hospital_name: str = None) -> bool:
    """
    Сохраняет запись в историю треков.
    Работает только если ENABLE_TRACKING_HISTORY = True
    """
    if not ENABLE_TRACKING_HISTORY:
        return False
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        bbox_json = json.dumps(bbox)
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT INTO tracking_history (person_id, track_id, session_id, timestamp, 
                                         bbox, confidence, camera_name, hospital_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (person_id, track_id, session_id, now, bbox_json, confidence, 
              camera_name, hospital_name))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        logging.error(f"Ошибка при сохранении истории треков: {e}")
        return False

def get_tracking_history(person_id: int, limit: int = 100) -> list:
    """
    Получает историю треков для конкретного человека.
    Работает только если ENABLE_TRACKING_HISTORY = True
    """
    if not ENABLE_TRACKING_HISTORY:
        return []
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT track_id, session_id, timestamp, bbox, confidence, camera_name, hospital_name
            FROM tracking_history 
            WHERE person_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (person_id, limit))
        history = cursor.fetchall()
        conn.close()
        return history
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении истории треков: {e}")
        return []

def get_session_tracking_history(session_id: str) -> list:
    """
    Получает историю треков для конкретной сессии.
    Работает только если ENABLE_TRACKING_HISTORY = True
    """
    if not ENABLE_TRACKING_HISTORY:
        return []
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT person_id, track_id, timestamp, bbox, confidence, camera_name, hospital_name
            FROM tracking_history 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        history = cursor.fetchall()
        conn.close()
        return history
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении истории треков сессии: {e}")
        return []

# === Утилитарные функции ===
def generate_session_id() -> str:
    """
    Генерирует уникальный идентификатор сессии.
    """
    return str(uuid.uuid4())

def get_statistics() -> dict:
    """
    Возвращает статистику по базе данных с учетом настроек.
    """
    stats = {}
    
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Количество лиц (если включено)
        if ENABLE_FACE_RECOGNITION:
            cursor.execute("SELECT COUNT(*) FROM faces")
            stats['faces_count'] = cursor.fetchone()[0]
        else:
            stats['faces_count'] = 0
        
        # Количество людей (если включено)
        if ENABLE_PERSON_RECOGNITION:
            cursor.execute("SELECT COUNT(*) FROM persons")
            stats['persons_count'] = cursor.fetchone()[0]
        else:
            stats['persons_count'] = 0
        
        # Количество связей (если включено)
        if ENABLE_FACE_PERSON_LINKING:
            cursor.execute("SELECT COUNT(*) FROM face_person_links")
            stats['links_count'] = cursor.fetchone()[0]
        else:
            stats['links_count'] = 0
        
        # Количество записей в истории (если включено)
        if ENABLE_TRACKING_HISTORY:
            cursor.execute("SELECT COUNT(*) FROM tracking_history")
            stats['history_count'] = cursor.fetchone()[0]
        else:
            stats['history_count'] = 0
        
        # Количество уникальных сессий (если включено)
        if ENABLE_PERSON_RECOGNITION:
            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM persons")
            stats['sessions_count'] = cursor.fetchone()[0]
        else:
            stats['sessions_count'] = 0
        
        conn.close()
        
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении статистики: {e}")
        stats = {
            'faces_count': 0,
            'persons_count': 0,
            'links_count': 0,
            'history_count': 0,
            'sessions_count': 0
        }
    
    return stats

def set_face_name(face_id: int, name: str):
    """
    Устанавливает или обновляет имя для лица по face_id.
    После установки имени автоматически объединяет все лица с этим именем.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE faces SET name = ? WHERE face_id = ?", (name, face_id))
        conn.commit()
        conn.close()
        
        # Автоматически объединяем лица с этим именем
        merge_faces_by_name_with_embedding(name)
        
    except Exception as e:
        logging.error(f'Ошибка при обновлении имени лица: {e}')

def get_face_name(face_id: int):
    """
    Получает имя лица по face_id.
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM faces WHERE face_id = ?", (face_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception as e:
        logging.error(f'Ошибка при получении имени лица: {e}')
        return None

def update_settings_flags():
    """
    Обновляет флаги настроек из базы данных в реальном времени.
    Вызывается при изменении настроек через UI.
    """
    try:
        import config
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Сохраняем текущие настройки в базу данных
        cursor.execute("DELETE FROM settings")
        cursor.execute("""
            INSERT INTO settings (frame_interval, hospital_name, confidence_threshold,
                                enable_face_recognition, enable_person_recognition, 
                                enable_face_person_linking, enable_tracking_history, 
                                enable_persistent_storage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            4,  # frame_interval по умолчанию
            "Не указано",  # hospital_name по умолчанию
            50,  # confidence_threshold по умолчанию
            config.ENABLE_FACE_RECOGNITION,
            config.ENABLE_PERSON_RECOGNITION,
            config.ENABLE_FACE_PERSON_LINKING,
            config.ENABLE_TRACKING_HISTORY,
            config.ENABLE_PERSISTENT_STORAGE
        ))
        conn.commit()
        
        # Обновляем глобальные переменные в этом модуле
        global ENABLE_FACE_RECOGNITION, ENABLE_PERSON_RECOGNITION, ENABLE_FACE_PERSON_LINKING
        global ENABLE_TRACKING_HISTORY, ENABLE_PERSISTENT_STORAGE
        
        ENABLE_FACE_RECOGNITION = config.ENABLE_FACE_RECOGNITION
        ENABLE_PERSON_RECOGNITION = config.ENABLE_PERSON_RECOGNITION
        ENABLE_FACE_PERSON_LINKING = config.ENABLE_FACE_PERSON_LINKING
        ENABLE_TRACKING_HISTORY = config.ENABLE_TRACKING_HISTORY
        ENABLE_PERSISTENT_STORAGE = config.ENABLE_PERSISTENT_STORAGE
        
        print(f"[Database] Настройки обновлены и сохранены: Face={ENABLE_FACE_RECOGNITION}, Person={ENABLE_PERSON_RECOGNITION}, Linking={ENABLE_FACE_PERSON_LINKING}")
        
        conn.close()
        
    except Exception as e:
        print(f"Ошибка при обновлении настроек: {e}")

def get_face_by_tracking_id(track_id, session_id=None):
    """
    Получает ID лица по ID трекинга.
    
    Args:
        track_id: ID трекинга от StrongSORT
        session_id: ID сессии (опционально)
        
    Returns:
        int: ID лица или None
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        if session_id:
            # Ищем person_id по track_id в текущей сессии
            cursor.execute("""
                SELECT person_id FROM persons 
                WHERE track_id = ? AND session_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, (track_id, session_id))
        else:
            # Ищем последний person_id по track_id
            cursor.execute("""
                SELECT person_id FROM persons 
                WHERE track_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, (track_id,))
        
        result = cursor.fetchone()
        if result:
            person_id = result[0]
            
            # Теперь ищем лицо по person_id
            cursor.execute("""
                SELECT face_id FROM face_person_links 
                WHERE person_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, (person_id,))
            
            face_result = cursor.fetchone()
            if face_result:
                conn.close()
                return face_result[0]
        
        conn.close()
        return None
        
    except sqlite3.Error as e:
        print(f"Ошибка при получении лица по треку: {e}")
        return None

def merge_faces_by_name_with_embedding(name: str) -> bool:
    """
    Объединяет все лица с одинаковым именем в одно лицо с усреднённым эмбеддингом.
    Оставляет лицо с наименьшим ID, остальные удаляет.
    Переносит все связи и обновляет эмбеддинг.
    """
    import numpy as np
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        # Получаем все лица с этим именем
        cursor.execute("SELECT face_id, embedding, first_seen, last_seen FROM faces WHERE name = ? ORDER BY face_id", (name,))
        faces = cursor.fetchall()
        if len(faces) < 2:
            conn.close()
            return True  # Нечего объединять
        main_face_id = faces[0][0]
        all_first_seen = [face[2] for face in faces]
        all_last_seen = [face[3] for face in faces]
        earliest_seen = min(all_first_seen)
        latest_seen = max(all_last_seen)
        # Собираем эмбеддинги
        embeddings = [np.frombuffer(face[1], dtype=np.float32) for face in faces]
        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding_bytes = mean_embedding.astype(np.float32).tobytes()
        # Обновляем эмбеддинг и временные метки основного лица
        cursor.execute("UPDATE faces SET embedding = ?, first_seen = ?, last_seen = ? WHERE face_id = ?", (mean_embedding_bytes, earliest_seen, latest_seen, main_face_id))
        # Переносим связи и удаляем остальные лица
        faces_to_delete = [face[0] for face in faces[1:]]
        for face_id in faces_to_delete:
            cursor.execute("UPDATE face_person_links SET face_id = ? WHERE face_id = ?", (main_face_id, face_id))
        if faces_to_delete:
            placeholders = ','.join(['?'] * len(faces_to_delete))
            cursor.execute(f"DELETE FROM faces WHERE face_id IN ({placeholders})", faces_to_delete)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        import logging
        logging.error(f'Ошибка при объединении лиц: {e}')
        return False

def auto_merge_faces_by_names() -> dict:
    """
    Автоматически объединяет все лица с одинаковыми именами.
    Возвращает словарь с результатами: {имя: количество_объединённых_лиц}
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Получаем все имена, которые встречаются более одного раза
        cursor.execute("""
            SELECT name, COUNT(*) as count 
            FROM faces 
            WHERE name IS NOT NULL AND name != '' 
            GROUP BY name 
            HAVING COUNT(*) > 1
        """)
        names_to_merge = cursor.fetchall()
        conn.close()
        
        results = {}
        for name, count in names_to_merge:
            if merge_faces_by_name_with_embedding(name):
                results[name] = count
        
        return results
    except Exception as e:
        import logging
        logging.error(f'Ошибка при автоматическом объединении лиц: {e}')
        return {}

def get_faces_with_names() -> list:
    """
    Получает список всех лиц с именами для ручного объединения.
    Возвращает список кортежей: (face_id, name, first_seen, last_seen)
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT face_id, name, first_seen, last_seen 
            FROM faces 
            WHERE name IS NOT NULL AND name != '' 
            ORDER BY name, face_id
        """)
        faces = cursor.fetchall()
        conn.close()
        return faces
    except Exception as e:
        import logging
        logging.error(f'Ошибка при получении лиц с именами: {e}')
        return []

def get_faces_by_name(name: str) -> list:
    """
    Получает список всех лиц с указанным именем.
    Возвращает список кортежей: (face_id, embedding, first_seen, last_seen)
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT face_id, embedding, first_seen, last_seen 
            FROM faces 
            WHERE name = ? 
            ORDER BY face_id
        """, (name,))
        faces = cursor.fetchall()
        conn.close()
        return faces
    except Exception as e:
        import logging
        logging.error(f'Ошибка при получении лиц по имени: {e}')
        return []

def clear_all_embeddings():
    """
    Очищает все эмбеддинги лиц и людей из базы данных.
    Удаляет данные из таблиц: faces, persons, face_person_links, tracking_history
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Очищаем таблицы в правильном порядке (сначала зависимые)
        tables_to_clear = [
            'tracking_history',
            'face_person_links', 
            'persons',
            'faces'
        ]
        
        deleted_counts = {}
        for table in tables_to_clear:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            cursor.execute(f"DELETE FROM {table}")
            deleted_counts[table] = count
        
        # Сбрасываем автоинкремент для всех таблиц
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('faces', 'persons', 'face_person_links', 'tracking_history')")
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'deleted_counts': deleted_counts,
            'total_deleted': sum(deleted_counts.values())
        }
        
    except sqlite3.Error as e:
        logging.error(f"Ошибка при очистке эмбеддингов: {e}")
        return {
            'success': False,
            'error': str(e),
            'deleted_counts': {},
            'total_deleted': 0
        }

def get_embeddings_statistics():
    """
    Возвращает статистику по эмбеддингам в базе данных
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        stats = {}
        
        # Статистика по лицам
        cursor.execute("SELECT COUNT(*) FROM faces")
        stats['faces'] = cursor.fetchone()[0]
        
        # Статистика по людям
        cursor.execute("SELECT COUNT(*) FROM persons")
        stats['persons'] = cursor.fetchone()[0]
        
        # Статистика по связям лиц и людей
        cursor.execute("SELECT COUNT(*) FROM face_person_links")
        stats['face_person_links'] = cursor.fetchone()[0]
        
        # Статистика по истории треков
        cursor.execute("SELECT COUNT(*) FROM tracking_history")
        stats['tracking_history'] = cursor.fetchone()[0]
        
        # Общее количество
        stats['total'] = stats['faces'] + stats['persons'] + stats['face_person_links'] + stats['tracking_history']
        
        conn.close()
        return stats
        
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении статистики эмбеддингов: {e}")
        return None

# === Функции для работы с фотороботами ===

def find_closest_face_with_distance(embedding_bytes: bytes, threshold: float = 0.66) -> dict:
    """
    Находит ближайшее лицо в базе данных по эмбеддингу и возвращает информацию о совпадении.
    
    Args:
        embedding_bytes: Эмбеддинг фоторобота
        threshold: Порог для определения совпадения
        
    Returns:
        dict: {
            'face_id': int,
            'name': str,
            'distance': float,
            'confidence': float,
            'first_seen': str,
            'last_seen': str
        } или None если совпадений нет
    """
    if not ENABLE_FACE_RECOGNITION or not ENABLE_PERSISTENT_STORAGE:
        return None
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT face_id, embedding, name, first_seen, last_seen 
            FROM faces 
            WHERE name IS NOT NULL AND name != ''
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        query_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        min_distance = float('inf')
        closest_match = None

        for face_id, stored_embedding_bytes, name, first_seen, last_seen in rows:
            stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)
            # Нормализуем векторы для косинусного расстояния
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
            # Вычисляем косинусное расстояние
            distance = 1 - np.dot(query_embedding_norm, stored_embedding_norm)
            
            if distance < min_distance:
                min_distance = distance
                closest_match = {
                    'face_id': face_id,
                    'name': name,
                    'distance': distance,
                    'confidence': 1 - distance,  # Уверенность = 1 - расстояние
                    'first_seen': first_seen,
                    'last_seen': last_seen
                }

        # Возвращаем результат только если расстояние меньше порога
        if closest_match and closest_match['distance'] <= threshold:
            return closest_match
        else:
            return None
            
    except Exception as e:
        logging.error(f"Ошибка при поиске ближайшего лица: {e}")
        return None

def find_top_matches_for_photobot(embedding_bytes: bytes, top_k: int = 5, threshold: float = 0.8) -> list:
    """
    Находит топ-K ближайших совпадений для фоторобота.
    
    Args:
        embedding_bytes: Эмбеддинг фоторобота
        top_k: Количество лучших совпадений
        threshold: Минимальный порог уверенности
        
    Returns:
        list: Список словарей с информацией о совпадениях, отсортированный по уверенности
    """
    if not ENABLE_FACE_RECOGNITION or not ENABLE_PERSISTENT_STORAGE:
        return []
        
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT face_id, embedding, name, first_seen, last_seen 
            FROM faces 
            WHERE name IS NOT NULL AND name != ''
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        query_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        matches = []

        for face_id, stored_embedding_bytes, name, first_seen, last_seen in rows:
            stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)
            # Нормализуем векторы для косинусного расстояния
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
            # Вычисляем косинусное расстояние
            distance = 1 - np.dot(query_embedding_norm, stored_embedding_norm)
            confidence = 1 - distance
            
            if confidence >= threshold:
                matches.append({
                    'face_id': face_id,
                    'name': name,
                    'distance': distance,
                    'confidence': confidence,
                    'first_seen': first_seen,
                    'last_seen': last_seen
                })

        # Сортируем по уверенности (по убыванию) и возвращаем топ-K
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches[:top_k]
        
    except Exception as e:
        logging.error(f"Ошибка при поиске топ совпадений: {e}")
        return []

def save_photobot_search_result(photobot_id: str, embedding_bytes: bytes, 
                               search_results: list, search_timestamp: str = None) -> bool:
    """
    Сохраняет результат поиска фоторобота в базу данных.
    
    Args:
        photobot_id: Уникальный ID фоторобота
        embedding_bytes: Эмбеддинг фоторобота
        search_results: Результаты поиска
        search_timestamp: Время поиска
        
    Returns:
        bool: True если сохранение успешно
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Создаем таблицу для истории поиска фотороботов если её нет
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photobot_search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photobot_id TEXT NOT NULL,
                photobot_embedding BLOB NOT NULL,
                search_timestamp TEXT NOT NULL,
                top_match_face_id INTEGER,
                top_match_name TEXT,
                top_match_confidence REAL,
                all_matches_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Подготавливаем данные для сохранения
        top_match = search_results[0] if search_results else None
        all_matches_json = json.dumps(search_results, ensure_ascii=False)
        
        cursor.execute("""
            INSERT INTO photobot_search_history 
            (photobot_id, photobot_embedding, search_timestamp, 
             top_match_face_id, top_match_name, top_match_confidence, all_matches_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            photobot_id,
            embedding_bytes,
            search_timestamp or datetime.now().isoformat(),
            top_match['face_id'] if top_match else None,
            top_match['name'] if top_match else None,
            top_match['confidence'] if top_match else None,
            all_matches_json
        ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"Ошибка при сохранении результата поиска фоторобота: {e}")
        return False

def get_photobot_search_history(limit: int = 50) -> list:
    """
    Получает историю поиска фотороботов.
    
    Args:
        limit: Максимальное количество записей
        
    Returns:
        list: Список записей истории поиска
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, photobot_id, search_timestamp, top_match_name, 
                   top_match_confidence, created_at
            FROM photobot_search_history 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'photobot_id': row[1],
                'search_timestamp': row[2],
                'top_match_name': row[3],
                'top_match_confidence': row[4],
                'created_at': row[5]
            }
            for row in rows
        ]
        
    except Exception as e:
        logging.error(f"Ошибка при получении истории поиска фотороботов: {e}")
        return []

def get_photobot_search_details(search_id: int) -> dict:
    """
    Получает детальную информацию о конкретном поиске фоторобота.
    
    Args:
        search_id: ID записи поиска
        
    Returns:
        dict: Детальная информация о поиске
    """
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT photobot_id, search_timestamp, top_match_face_id, 
                   top_match_name, top_match_confidence, all_matches_json, created_at
            FROM photobot_search_history 
            WHERE id = ?
        """, (search_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'photobot_id': row[0],
                'search_timestamp': row[1],
                'top_match_face_id': row[2],
                'top_match_name': row[3],
                'top_match_confidence': row[4],
                'all_matches': json.loads(row[5]) if row[5] else [],
                'created_at': row[6]
            }
        else:
            return None
            
    except Exception as e:
        logging.error(f"Ошибка при получении деталей поиска фоторобота: {e}")
        return None



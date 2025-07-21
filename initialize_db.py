# initialize_db.py

import sqlite3
from config import DATABASE_FILE

def initialize_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Таблица для камер
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cameras (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        ip TEXT,
        port TEXT,
        username TEXT,
        password TEXT,
        protocol TEXT,
        path TEXT,
        max_people_count INTEGER
    )
    """)
    
    # Таблица для событий превышения лимита в видео
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_exceedance_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT,
        event_time TEXT,
        people_count INTEGER,
        max_people_count INTEGER
    )
    """)
    
    # Таблица для логов ошибок
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS error_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        error_message TEXT,
        camera_id INTEGER,
        camera_name TEXT,
        stack_trace TEXT
    )
    """)
    
    # Таблица для периодической статистики
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS periodic_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        camera_name TEXT,
        hospital_name TEXT,
        people_count INTEGER
    )
    """)
    
    # Таблица для записей о превышении количества людей
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS people_exceeded (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_name TEXT,
        hospital_name TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        people_count INTEGER,
        max_people_count INTEGER
    )
    """)
    
    # Таблица для настроек
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_interval INTEGER,
        hospital_name TEXT,
        confidence_threshold INTEGER,
        photo_data TEXT,
        enable_face_recognition BOOLEAN DEFAULT 1,
        enable_person_recognition BOOLEAN DEFAULT 1,
        enable_face_person_linking BOOLEAN DEFAULT 1,
        enable_tracking_history BOOLEAN DEFAULT 1,
        enable_persistent_storage BOOLEAN DEFAULT 1,
        face_confidence_threshold INTEGER DEFAULT 80
    )
    """)
    
    # Добавляем поле face_confidence_threshold, если его нет
    try:
        cursor.execute("ALTER TABLE settings ADD COLUMN face_confidence_threshold INTEGER DEFAULT 80")
        print("✅ Добавлено поле face_confidence_threshold в таблицу settings")
    except sqlite3.OperationalError:
        # Поле уже существует
        pass
    
    # Таблица для пользователей Telegram
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        hospital_name TEXT NOT NULL,
        registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(chat_id, hospital_name)
    )
    """)
    
    # === ТАБЛИЦЫ ДЛЯ РАСПОЗНАВАНИЯ ЛИЦ И ЛЮДЕЙ ===
    
    # Таблица для лиц
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        face_id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding BLOB NOT NULL,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        face_bbox TEXT,  -- JSON: {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        person_id INTEGER,
        name TEXT
    )
    """)
    
    # Таблица для людей
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        person_id INTEGER PRIMARY KEY AUTOINCREMENT,
        reid_embedding BLOB NOT NULL,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        person_bbox TEXT,  -- JSON: {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        track_id INTEGER,  -- ID трека в рамках сессии
        session_id TEXT,   -- Идентификатор сессии
        camera_name TEXT,
        hospital_name TEXT,
        name TEXT
    )
    """)
    
    # Таблица для связей лиц и людей
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_person_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_id INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        confidence REAL DEFAULT 0.0,
        linked_at TEXT NOT NULL,
        UNIQUE(face_id, person_id)
    )
    """)
    
    # Таблица для истории треков
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tracking_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        track_id INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        bbox TEXT NOT NULL,  -- JSON: {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        confidence REAL DEFAULT 0.0,
        camera_name TEXT,
        hospital_name TEXT
    )
    """)
    
    # Создаем индексы для быстрого поиска
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_embedding ON faces(embedding)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_persons_reid_embedding ON persons(reid_embedding)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_persons_session ON persons(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_persons_track ON persons(track_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_person_links ON face_person_links(face_id, person_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracking_history_person ON tracking_history(person_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracking_history_session ON tracking_history(session_id)")
    
    conn.commit()
    conn.close()
    print("✅ База данных инициализирована успешно!")
    print("📋 Созданные таблицы:")
    print("   - cameras (камеры)")
    print("   - settings (настройки)")
    print("   - video_exceedance_stats (статистика превышений видео)")
    print("   - error_logs (логи ошибок)")
    print("   - periodic_stats (периодическая статистика)")
    print("   - people_exceeded (превышения лимита людей)")
    print("   - users (пользователи Telegram)")
    print("   - faces (распознавание лиц)")
    print("   - persons (распознавание людей)")
    print("   - face_person_links (связи лиц и людей)")
    print("   - tracking_history (история треков)")

if __name__ == '__main__':
    initialize_database()

# migrate_persons.py

import sqlite3
from config import DATABASE_FILE

def migrate_db():
    """Миграция базы данных под новую структуру persons и индексы."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        print("Начинаем миграцию базы данных...")
        
        # 1. Создаем новую таблицу persons с нужными полями
        print("1. Создание/обновление таблицы persons...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                reid_embedding BLOB NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                person_bbox TEXT,
                track_id INTEGER,
                session_id TEXT,
                camera_name TEXT,
                hospital_name TEXT
            )
        ''')
        
        # 2. Создаем индексы для persons
        print("2. Создание индексов для persons...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_persons_reid_embedding ON persons(reid_embedding)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_persons_session ON persons(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_persons_track ON persons(track_id)")
        
        # 3. Создаем таблицу face_person_links
        print("3. Создание таблицы face_person_links...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_person_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                confidence REAL DEFAULT 0.0,
                linked_at TEXT NOT NULL,
                UNIQUE(face_id, person_id)
            )
        ''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_person_links ON face_person_links(face_id, person_id)")
        
        # 4. Создаем таблицу tracking_history
        print("4. Создание таблицы tracking_history...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                bbox TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                camera_name TEXT,
                hospital_name TEXT
            )
        ''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracking_history_person ON tracking_history(person_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracking_history_session ON tracking_history(session_id)")
        
        conn.commit()
        conn.close()
        print("✅ Миграция завершена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при миграции: {e}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    migrate_db() 
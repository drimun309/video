import sqlite3
import os

def check_cameras_db():
    db_path = 'cameras.db'
    if not os.path.exists(db_path):
        print(f"❌ Файл {db_path} не найден")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Проверяем таблицы
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"📋 Таблицы в базе данных: {tables}")
        
        # Проверяем таблицу cameras
        if ('cameras',) in tables:
            cursor.execute("SELECT COUNT(*) FROM cameras")
            count = cursor.fetchone()[0]
            print(f"📹 Количество камер в базе: {count}")
            
            if count > 0:
                cursor.execute("SELECT id, name, ip, port, protocol FROM cameras LIMIT 3")
                cameras = cursor.fetchall()
                print("📹 Примеры камер:")
                for camera in cameras:
                    print(f"  ID: {camera[0]}, Имя: {camera[1]}, IP: {camera[2]}:{camera[3]}, Протокол: {camera[4]}")
        else:
            print("❌ Таблица 'cameras' не найдена")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка при работе с базой данных: {e}")

if __name__ == "__main__":
    check_cameras_db() 
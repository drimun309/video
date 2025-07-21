# quick_clear_embeddings.py
"""
Быстрый скрипт для очистки базы данных от эмбеддингов без подтверждения.
Используется для автоматизации или быстрой очистки.
"""

import sqlite3
import os
from datetime import datetime
from config import DATABASE_FILE

def quick_clear_embeddings():
    """Быстрая очистка всех эмбеддингов без подтверждения"""
    try:
        if not os.path.exists(DATABASE_FILE):
            print(f"❌ База данных {DATABASE_FILE} не найдена!")
            return False
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        print(f"🧹 Быстрая очистка эмбеддингов: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Очищаем таблицы в правильном порядке
        tables_to_clear = [
            'tracking_history',
            'face_person_links', 
            'persons',
            'faces'
        ]
        
        total_deleted = 0
        for table in tables_to_clear:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            cursor.execute(f"DELETE FROM {table}")
            total_deleted += count
            print(f"   ✅ {table}: {count} записей удалено")
        
        # Сбрасываем автоинкремент
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('faces', 'persons', 'face_person_links', 'tracking_history')")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Очистка завершена! Всего удалено: {total_deleted} записей")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при очистке: {e}")
        return False

def get_embeddings_count():
    """Возвращает количество записей с эмбеддингами"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM faces")
        faces_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM persons")
        persons_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM face_person_links")
        links_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tracking_history")
        history_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'faces': faces_count,
            'persons': persons_count,
            'links': links_count,
            'history': history_count,
            'total': faces_count + persons_count + links_count + history_count
        }
    except Exception as e:
        print(f"❌ Ошибка при подсчете: {e}")
        return None

if __name__ == '__main__':
    # Показываем статистику до очистки
    stats = get_embeddings_count()
    if stats:
        print(f"📊 До очистки: {stats['total']} записей")
        print(f"   👥 Лица: {stats['faces']}")
        print(f"   🚶 Люди: {stats['persons']}")
        print(f"   🔗 Связи: {stats['links']}")
        print(f"   📈 История: {stats['history']}")
    
    # Выполняем очистку
    success = quick_clear_embeddings()
    
    if success:
        # Проверяем результат
        stats_after = get_embeddings_count()
        if stats_after and stats_after['total'] == 0:
            print("🎉 База данных успешно очищена от всех эмбеддингов!")
        else:
            print("⚠️  Очистка завершена, но некоторые записи могут остаться.")
    else:
        print("❌ Очистка не удалась.") 
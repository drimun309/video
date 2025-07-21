# network.py

import os
import requests
from datetime import datetime
from config import SERVER_URL
from PyQt6.QtCore import QThread, pyqtSignal
import json
from config import SERVER_URL
import network  # если нужно для глобальной переменной
from config import DATABASE_FILE
# Глобальная переменная для хранения статуса сервера
SERVER_ENABLED = False




def is_server_active():
    """
    Отправляет запрос на сервер для проверки его доступности.
    Предполагается, что на сервере реализован эндпоинт /ping, который возвращает 200 OK.
    Функция обновляет глобальную переменную SERVER_ENABLED.
    """
    global SERVER_ENABLED
    try:
        response = requests.get(f"{SERVER_URL}/ping", timeout=3)
        if response.status_code == 200:
            SERVER_ENABLED = True
            return True
        else:
            SERVER_ENABLED = False
            return False
    except requests.exceptions.RequestException:
        SERVER_ENABLED = False
        return False
    
def send_exceeded_stats(hospital_name, camera_name, people_count, max_people_count):
    """
    Отправляет exceeded‑статистику (без фото) на сервер через POST‑запрос.
    """
    try:
        url = f"{SERVER_URL}/api/exceeded"
        data = {
            "camera_name": camera_name,
            "hospital_name": hospital_name,
            "people_count": people_count,
            "max_people_count": max_people_count,
            "timestamp": datetime.now().isoformat()
        }
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 201:
            print(f"Exceeded-статистика для камеры '{camera_name}' успешно отправлена")
            return True
        else:
            print(f"Ошибка отправки exceeded stats: {response.status_code}, {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения с сервером при отправке exceeded stats: {e}")
        return False


def send_stats_to_server(stats_data):
    print("send_stats_to_server вызвана с данными:", json.dumps(stats_data, indent=2))
    
    global SERVER_ENABLED
    if not SERVER_ENABLED:
        print("SERVER_DISABLED: Пропускаем отправку статистики.")
        return True
    try:
        response = requests.post(f"{SERVER_URL}/api/upload_stats", json=stats_data, timeout=10)
        print("Ответ сервера:", response.status_code, response.text)
        if response.status_code == 200:
            print("Статистика успешно отправлена на сервер")
            return True
        else:
            print(f"Ошибка отправки статистики: {response.status_code}, {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения с сервером: {e}")
        return False


    

def send_exceeded_stats_with_photo(hospital_name, camera_name, people_count, max_people_count, photo_data):
    """
    Отправляет exceeded‑статистику с фото (в base64) на сервер через POST‑запрос.
    Если фото не требуется, photo_data может быть None.
    """
    if not SERVER_ENABLED:
        print("SERVER_DISABLED: Пропускаем отправку exceeded stats.")
        return True
    try:
        url = f"{SERVER_URL}/api/exceeded"
        data = {
            "camera_name": camera_name,
            "hospital_name": hospital_name,
            "people_count": people_count,
            "max_people_count": max_people_count,
            "timestamp": datetime.now().isoformat(),
            "photo_data": photo_data
        }
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 201:
            print(f"Статистика о превышении для камеры '{camera_name}' успешно отправлена")
            return True
        else:
            print(f"Ошибка отправки exceeded stats: {response.status_code}, {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения с сервером при отправке exceeded stats: {e}")
        return False

def send_exceeded_stats_async(hospital_name, camera_name, people_count, max_people_count):
    """
    Асинхронно отправляет статистику превышений на сервер.
    """
    def send():
        try:
            url = f"{SERVER_URL}/api/exceeded"
            data = {
                "camera_name": camera_name,
                "hospital_name": hospital_name,
                "people_count": people_count,
                "max_people_count": max_people_count,
                "timestamp": datetime.now().isoformat()
            }
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 201:
                print(f"[DEBUG] Статистика превышения отправлена: {camera_name}, людей={people_count}")
                return True
            else:
                print(f"Ошибка отправки: {response.status_code}, {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Ошибка соединения с сервером: {e}")
            return False

    worker = NetworkWorker(send)
    worker.start()
    return worker


def send_screenshot_to_server(hospital_name, camera_name, screenshot_path):
    """
    Отправляет скриншот на сервер через POST запрос.
    """
    global SERVER_ENABLED
    if not SERVER_ENABLED:
        print("SERVER_DISABLED: Пропускаем отправку скриншота.")
        return True
    try:
        url = f"{SERVER_URL}/upload_snapshot/{hospital_name}"
        with open(screenshot_path, 'rb') as file:
            files = {'file': (os.path.basename(screenshot_path), file)}
            response = requests.post(url, files=files, timeout=10)
        if response.status_code == 200:
            print(f"Скриншот для камеры '{camera_name}' успешно отправлен")
            return True
        else:
            print(f"Ошибка отправки скриншота: {response.status_code}, {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения с сервером при отправке скриншота: {e}")
        return False

class NetworkWorker(QThread):
    finished_signal = pyqtSignal(object)
    
    def __init__(self, func, *args, **kwargs):
        parent = kwargs.pop("parent", None)
        super().__init__(parent)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        print("NetworkWorker initialized with function:", func.__name__)
    
    def run(self):
        print("NetworkWorker running")
        try:
            result = self.func(*self.args, **self.kwargs)
            print("NetworkWorker finished, result:", result)
            self.finished_signal.emit(result)
        except Exception as e:
            print("NetworkWorker encountered exception:", e)
            self.finished_signal.emit(e)



class ServerCheckThread(QThread):
    """
    Поток для проверки статуса сервера.
    По завершении работы отправляет результат через result_signal.
    """
    result_signal = pyqtSignal(bool)

    def run(self):
        active = is_server_active()
        self.result_signal.emit(active)

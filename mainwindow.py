# mainwindow.py

import sys
import time
import traceback
import sqlite3
import config

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QPushButton, QLabel, QComboBox, QLineEdit, QFormLayout, QDialog, QMessageBox, QTableWidget,
                             QTableWidgetItem, QScrollArea, QSplitter, QSizePolicy, QInputDialog, QCheckBox)
from PyQt6.QtGui import QIcon, QPixmap

from config import START_ICON_PATH, STOP_ICON_PATH, DEFAULT_FRAME_INTERVAL, DEFAULT_CONFIDENCE_THRESHOLD, DATABASE_FILE, FACE_THRESHOLD
from config import ENABLE_FACE_RECOGNITION, ENABLE_PERSON_RECOGNITION, ENABLE_FACE_PERSON_LINKING, ENABLE_TRACKING_HISTORY, ENABLE_PERSISTENT_STORAGE
from database import load_cameras, save_cameras, execute_db_query, insert_error_log, get_faces_count, set_face_name, update_settings_flags
from network import is_server_active, ServerCheckThread
from video_processing import VideoProcessingThread, CameraThread, ROISelector, VideoRecorder
from helpers import parse_video_filename, format_datetime
from dialogs import ROISelectorDialog, SetFaceNameDialog, ManualMergeFacesDialog, AutoMergeFacesDialog, PhotobotSearchDialog, PhotobotHistoryDialog


# Простейшие реализации диалоговых окон для добавления и редактирования камер.
class AddCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Добавить Камеру")
        self.layout = QFormLayout(self)
        self.name_input = QLineEdit()
        self.ip_input = QLineEdit()
        self.port_input = QLineEdit()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.protocol_input = QLineEdit()
        self.path_input = QLineEdit()
        self.max_people_input = QLineEdit()
        self.layout.addRow("Имя камеры:", self.name_input)
        self.layout.addRow("IP:", self.ip_input)
        self.layout.addRow("Порт:", self.port_input)
        self.layout.addRow("Имя пользователя:", self.username_input)
        self.layout.addRow("Пароль:", self.password_input)
        self.layout.addRow("Протокол:", self.protocol_input)
        self.layout.addRow("Путь:", self.path_input)
        self.layout.addRow("Макс. людей:", self.max_people_input)
        btn = QPushButton("Добавить")
        btn.clicked.connect(self.accept)
        self.layout.addRow(btn)

    def get_camera_data(self):
        return {
            "name": self.name_input.text(),
            "ip": self.ip_input.text(),
            "port": self.port_input.text(),
            "username": self.username_input.text(),
            "password": self.password_input.text(),
            "protocol": self.protocol_input.text(),
            "path": self.path_input.text(),
            "max_people_count": int(self.max_people_input.text() or 0)
        }

class CameraDetailsDialog(QDialog):
    def __init__(self, camera_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Детали камеры")
        self.layout = QFormLayout(self)
        self.name_input = QLineEdit(camera_data.get("name", ""))
        self.ip_input = QLineEdit(camera_data.get("ip", ""))
        self.port_input = QLineEdit(str(camera_data.get("port", "")))
        self.username_input = QLineEdit(camera_data.get("username", ""))
        self.password_input = QLineEdit(camera_data.get("password", ""))
        self.protocol_input = QLineEdit(camera_data.get("protocol", ""))
        self.path_input = QLineEdit(camera_data.get("path", ""))
        self.max_people_input = QLineEdit(str(camera_data.get("max_people_count", 0)))
        self.layout.addRow("Имя камеры:", self.name_input)
        self.layout.addRow("IP:", self.ip_input)
        self.layout.addRow("Порт:", self.port_input)
        self.layout.addRow("Имя пользователя:", self.username_input)
        self.layout.addRow("Пароль:", self.password_input)
        self.layout.addRow("Протокол:", self.protocol_input)
        self.layout.addRow("Путь:", self.path_input)
        self.layout.addRow("Макс. людей:", self.max_people_input)
        btn = QPushButton("Сохранить")
        btn.clicked.connect(self.accept)
        self.layout.addRow(btn)

    def get_updated_camera_data(self):
        return {
            "name": self.name_input.text(),
            "ip": self.ip_input.text(),
            "port": self.port_input.text(),
            "username": self.username_input.text(),
            "password": self.password_input.text(),
            "protocol": self.protocol_input.text(),
            "path": self.path_input.text(),
            "max_people_count": int(self.max_people_input.text() or 0)
        }

class MainWindow(QMainWindow):
    error_log_signal = pyqtSignal(str, int, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Камеры и Трансляция")
        self.resize(1200, 700)

        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.face_confidence_threshold = FACE_THRESHOLD
        self.frame_interval = DEFAULT_FRAME_INTERVAL

        # Загружаем камеры из базы данных
        cameras_list = load_cameras()
        self.cameras = {camera['id']: camera for camera in cameras_list}
        self.active_threads = {}
        self.active_cameras = []
        self.current_camera = None
        self.current_frame = None
        self.detection_roi = None
        self.video_list = []
        self.video_combo_box = QComboBox()
        self.video_combo_box.addItem("Выберите видео")
        self.video_combo_box.currentIndexChanged.connect(self.on_video_selection_changed)
        self.video_start = None
        self.video_session_start = None
        self.last_periodic_log_time = 0
        self.camera_ui_mapping = []
        
        # Переменные для записи видео
        self.current_recording_camera = None
        self.current_recording_video = None
        
        # Создаем статус бар
        self.statusBar().showMessage("Готов к работе")
        
        # Таймер для обновления статистики лиц
        self.face_stats_timer = QTimer()
        self.face_stats_timer.timeout.connect(self.update_face_stats)
        self.face_stats_timer.start(5000)  # Обновление каждые 5 секунд
        
        self.init_ui()
        self.load_saved_cameras()
        self.load_settings()

        self.error_log_signal.connect(self.handle_error)
        
        # Первоначальное обновление статистики лиц
        self.update_face_stats()

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_camera_tab(), "Камеры и Трансляция")
        self.tabs.addTab(self.create_settings_tab(), "Настройки")
        self.tabs.addTab(self.create_photobot_tab(), "Фотороботы")
        self.tabs.addTab(self.create_video_tab(), "Видео")
        main_layout.addWidget(self.tabs)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_camera_tab(self):
        tab = QWidget()
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Виджет трансляции
        self.stream_screen = QLabel("Трансляция не выбрана")
        self.stream_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stream_screen.setMinimumSize(640, 480)
        splitter.addWidget(self.stream_screen)
        
        # Панель управления
        control_widget = QWidget()
        control_layout = QVBoxLayout()

        add_camera_button = QPushButton("Добавить Камеру")
        add_camera_button.clicked.connect(self.open_add_camera_dialog)
        control_layout.addWidget(add_camera_button)

        select_roi_button = QPushButton("Выделить область")
        select_roi_button.clicked.connect(self.select_roi)
        control_layout.addWidget(select_roi_button)

        self.camera_layout = QVBoxLayout()
        control_layout.addLayout(self.camera_layout)

        self.camera_select_combo = QComboBox()
        self.camera_select_combo.addItem("Выберите камеру для трансляции")
        self.camera_select_combo.currentIndexChanged.connect(self.select_camera_for_stream)
        control_layout.addWidget(self.camera_select_combo)

        set_face_name_button = QPushButton("Задать имя лицу")
        set_face_name_button.clicked.connect(self.open_set_face_name_dialog)
        control_layout.addWidget(set_face_name_button)

        merge_faces_button = QPushButton("Ручное объединение лиц")
        merge_faces_button.clicked.connect(self.open_manual_merge_faces_dialog)
        control_layout.addWidget(merge_faces_button)

        auto_merge_faces_button = QPushButton("Автоматическое объединение лиц")
        auto_merge_faces_button.clicked.connect(self.open_auto_merge_faces_dialog)
        control_layout.addWidget(auto_merge_faces_button)

        # === Кнопки записи видео ===
        recording_label = QLabel("Запись видео:")
        recording_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #2c3e50;")
        control_layout.addWidget(recording_label)
        
        self.camera_recording_button = QPushButton("Начать запись видео")
        self.camera_recording_button.setCheckable(True)
        self.camera_recording_button.clicked.connect(self.toggle_camera_recording)
        control_layout.addWidget(self.camera_recording_button)
        
        self.camera_recording_status_label = QLabel("Статус записи: Остановлена")
        self.camera_recording_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        control_layout.addWidget(self.camera_recording_status_label)

        # Удаляем разделитель и кнопки фотороботов из панели управления камер
        # (separator_label, photobot_search_button, photobot_history_button)

        control_widget.setLayout(control_layout)
        control_widget.setMinimumWidth(300)
        splitter.addWidget(control_widget)
        splitter.setSizes([800, 360])
        layout.addWidget(splitter)
        tab.setLayout(layout)
        return tab

    def create_settings_tab(self):
        tab = QWidget()
        layout = QFormLayout()
        
        # === Настройки детекции людей ===
        people_detection_label = QLabel("Настройки детекции людей:")
        people_detection_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #2c3e50;")
        layout.addRow(people_detection_label)
        
        self.confidence_input = QLineEdit("50")
        self.confidence_input.setPlaceholderText("Введите значение от 10 до 100")
        self.confidence_input.textChanged.connect(self.update_confidence_threshold)
        self.confidence_label = QLabel("Уверенность: 50%")
        layout.addRow("Уровень уверенности для детекции людей:", self.confidence_input)
        layout.addRow(self.confidence_label)

        # === Настройки детекции лиц ===
        face_detection_label = QLabel("Настройки детекции лиц:")
        face_detection_label.setStyleSheet("font-weight: bold; margin-top: 15px; color: #2c3e50;")
        layout.addRow(face_detection_label)
        
        self.face_confidence_input = QLineEdit(str(int(FACE_THRESHOLD * 100)))
        self.face_confidence_input.setPlaceholderText("Введите значение от 10 до 100")
        self.face_confidence_input.textChanged.connect(self.update_face_confidence_threshold)
        self.face_confidence_label = QLabel(f"Уверенность: {int(FACE_THRESHOLD * 100)}%")
        layout.addRow("Уровень уверенности для детекции лиц:", self.face_confidence_input)
        layout.addRow(self.face_confidence_label)

        # === Общие настройки ===
        general_settings_label = QLabel("Общие настройки:")
        general_settings_label.setStyleSheet("font-weight: bold; margin-top: 15px; color: #2c3e50;")
        layout.addRow(general_settings_label)

        self.frame_interval_combo = QComboBox()
        self.frame_interval_combo.addItems(["4", "8", "16", "20"])
        self.frame_interval_combo.setCurrentText("4")
        self.frame_interval_combo.currentTextChanged.connect(self.update_frame_interval)
        layout.addRow("Частота кадров для обработки:", self.frame_interval_combo)

        self.hospital_name_input = QLineEdit()
        self.hospital_name_input.setPlaceholderText("Введите название больницы")
        layout.addRow("Название больницы:", self.hospital_name_input)

        self.server_status_label = QLabel("Статус сервера: неизвестно")
        check_server_button = QPushButton("Проверить сервер")
        check_server_button.clicked.connect(self.check_server_status)
        layout.addRow("Статус сервера:", self.server_status_label)
        layout.addRow(check_server_button)

        # --- Чекбоксы для флагов ---
        self.cb_face_recognition = QCheckBox("Распознавание лиц (ENABLE_FACE_RECOGNITION)")
        self.cb_face_recognition.setChecked(ENABLE_FACE_RECOGNITION)
        self.cb_face_recognition.stateChanged.connect(self.update_face_recognition_setting)
        layout.addRow(self.cb_face_recognition)

        self.cb_person_recognition = QCheckBox("Распознавание людей (ENABLE_PERSON_RECOGNITION)")
        self.cb_person_recognition.setChecked(ENABLE_PERSON_RECOGNITION)
        self.cb_person_recognition.stateChanged.connect(self.update_person_recognition_setting)
        layout.addRow(self.cb_person_recognition)

        self.cb_face_person_linking = QCheckBox("Связывание лиц и людей (ENABLE_FACE_PERSON_LINKING)")
        self.cb_face_person_linking.setChecked(ENABLE_FACE_PERSON_LINKING)
        self.cb_face_person_linking.stateChanged.connect(self.update_face_person_linking_setting)
        layout.addRow(self.cb_face_person_linking)

        self.cb_tracking_history = QCheckBox("Сохранение истории треков (ENABLE_TRACKING_HISTORY)")
        self.cb_tracking_history.setChecked(ENABLE_TRACKING_HISTORY)
        self.cb_tracking_history.stateChanged.connect(self.update_tracking_history_setting)
        layout.addRow(self.cb_tracking_history)

        self.cb_persistent_storage = QCheckBox("Сохранение эмбеддингов в БД (ENABLE_PERSISTENT_STORAGE)")
        self.cb_persistent_storage.setChecked(ENABLE_PERSISTENT_STORAGE)
        self.cb_persistent_storage.stateChanged.connect(self.update_persistent_storage_setting)
        layout.addRow(self.cb_persistent_storage)
        # --- конец чекбоксов ---

        save_button = QPushButton("Сохранить Настройки")
        save_button.clicked.connect(self.save_settings)
        layout.addRow(save_button)
        tab.setLayout(layout)
        return tab

    def create_photobot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        search_button = QPushButton("Поиск совпадений фоторобота")
        search_button.clicked.connect(self.open_photobot_search_dialog)
        layout.addWidget(search_button)
        history_button = QPushButton("История поиска фотороботов")
        history_button.clicked.connect(self.open_photobot_history_dialog)
        layout.addWidget(history_button)
        layout.addStretch(1)
        tab.setLayout(layout)
        return tab

    def create_video_tab(self):
        tab = QWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.video_display_label = QLabel("Видео/Изображение не выбрано")
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_display_label.setMinimumSize(800, 600)
        splitter.addWidget(self.video_display_label)

        control_widget = QWidget()
        control_layout = QVBoxLayout()
        
        # Группа для загрузки файлов
        file_group_label = QLabel("Загрузка файлов:")
        file_group_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        control_layout.addWidget(file_group_label)
        
        load_video_button = QPushButton("Добавить видео")
        load_video_button.clicked.connect(self.add_video)
        control_layout.addWidget(load_video_button)
        
        self.video_combo_box = QComboBox()
        self.video_combo_box.addItem("Выберите видео")
        self.video_combo_box.currentIndexChanged.connect(self.on_video_selection_changed)
        control_layout.addWidget(self.video_combo_box)

        self.video_name_label = QLabel("Видео: не выбрано")
        control_layout.addWidget(self.video_name_label)

        self.video_toggle_button = QPushButton()
        self.video_toggle_button.setCheckable(True)
        self.video_toggle_button.setIcon(QIcon(START_ICON_PATH))
        self.video_toggle_button.clicked.connect(self.toggle_video)
        control_layout.addWidget(self.video_toggle_button)

        self.video_roi_button = QPushButton("Выбрать область детекции")
        self.video_roi_button.clicked.connect(self.select_video_roi)
        control_layout.addWidget(self.video_roi_button)

        self.video_count_label = QLabel("Людей: 0")
        control_layout.addWidget(self.video_count_label)

        # === Кнопки записи видео ===
        video_recording_label = QLabel("Запись видео:")
        video_recording_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #2c3e50;")
        control_layout.addWidget(video_recording_label)
        
        self.video_recording_button = QPushButton("Начать запись видео")
        self.video_recording_button.setCheckable(True)
        self.video_recording_button.clicked.connect(self.toggle_video_recording)
        control_layout.addWidget(self.video_recording_button)
        
        self.video_recording_status_label = QLabel("Статус записи: Остановлена")
        self.video_recording_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        control_layout.addWidget(self.video_recording_status_label)

        max_people_text = QLabel("Максимальное количество людей:")
        self.video_max_people_input = QLineEdit("0")
        control_layout.addWidget(max_people_text)
        control_layout.addWidget(self.video_max_people_input)

        # --- Кнопка для работы с людьми в видео ---
        # set_person_name_video_button = QPushButton("Задать имя человеку")
        # set_person_name_video_button.clicked.connect(self.open_set_person_name_dialog)
        # control_layout.addWidget(set_person_name_video_button)

        # --- Кнопка для задания имени лицу в видео ---
        set_face_name_video_button = QPushButton("Задать имя лицу")
        set_face_name_video_button.clicked.connect(self.open_set_face_name_dialog)
        control_layout.addWidget(set_face_name_video_button)

        # --- Кнопка для ручного объединения лиц в видео ---
        merge_faces_video_button = QPushButton("Ручное объединение лиц")
        merge_faces_video_button.clicked.connect(self.open_manual_merge_faces_dialog)
        control_layout.addWidget(merge_faces_video_button)

        # --- Кнопка для автоматического объединения лиц в видео ---
        auto_merge_faces_video_button = QPushButton("Автоматическое объединение лиц")
        auto_merge_faces_video_button.clicked.connect(self.open_auto_merge_faces_dialog)
        control_layout.addWidget(auto_merge_faces_video_button)



        control_layout.addStretch()
        control_widget.setLayout(control_layout)
        splitter.addWidget(control_widget)

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        delete_video_button = QPushButton("Удалить видео")
        delete_video_button.clicked.connect(self.delete_video)
        control_layout.addWidget(delete_video_button)

        tab.setLayout(layout)
        return tab

    def update_confidence_threshold(self, value):
        try:
            confidence_value = int(value)
            if 10 <= confidence_value <= 100:
                self.confidence_threshold = confidence_value / 100
                self.confidence_label.setText(f"Уверенность: {confidence_value}%")
                for thread in self.active_threads.values():
                    thread.confidence_threshold = self.confidence_threshold
            else:
                print("Ошибка: Значение уверенности должно быть от 10 до 100.")
        except ValueError:
            print("Ошибка: Некорректное значение уверенности.")

    def update_face_confidence_threshold(self, value):
        try:
            confidence_value = int(value)
            if 10 <= confidence_value <= 100:
                self.face_confidence_threshold = confidence_value / 100
                self.face_confidence_label.setText(f"Уверенность: {confidence_value}%")
                # Обновляем глобальную переменную в config
                import config
                config.FACE_THRESHOLD = self.face_confidence_threshold
                print(f"Порог уверенности для детекции лиц обновлен: {confidence_value}%")
            else:
                print("Ошибка: Значение уверенности должно быть от 10 до 100.")
        except ValueError:
            print("Ошибка: Некорректное значение уверенности.")

    def update_frame_interval(self, interval):
        self.frame_interval = int(interval)
        for thread in self.active_threads.values():
            thread.frame_interval = self.frame_interval

    def save_settings(self):
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM settings")
            cursor.execute("""
                INSERT INTO settings (frame_interval, hospital_name, confidence_threshold, face_confidence_threshold)
                VALUES (?, ?, ?, ?)
            """, (self.frame_interval, self.hospital_name_input.text(), 
                  int(self.confidence_threshold * 100), 
                  int(getattr(self, 'face_confidence_threshold', FACE_THRESHOLD) * 100)))
            conn.commit()
            conn.close()
            
            # Сохраняем настройку в config.py
            import config
            config.FACE_THRESHOLD = getattr(self, 'face_confidence_threshold', FACE_THRESHOLD)
            
            QMessageBox.information(self, "Успех", "Настройки успешно сохранены!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении настроек: {e}")


    def select_roi(self):
        # Проверяем, что активен какой-либо поток камеры
        if self.current_camera is None or self.current_camera not in self.active_threads:
            QMessageBox.warning(self, "Ошибка", "Нет активного потока для выбора области.")
            return

        thread = self.active_threads[self.current_camera]
        if thread.last_frame is None:
            QMessageBox.warning(self, "Ошибка", "Нет доступного кадра для выбора области.")
            return

        # Приостанавливаем обновление UI от камеры, чтобы кадр не менялся во время выбора
        thread.pause_updates = True

        # Импортируем диалог выбора ROI из модуля dialogs (убедитесь, что модуль dialogs.py находится в проекте)
        roi_dialog = ROISelectorDialog(frame=thread.last_frame, parent=self)
        if roi_dialog.exec() == QDialog.DialogCode.Accepted:
            roi = roi_dialog.get_roi()
            if roi is not None:
                thread.roi = roi
                QMessageBox.information(self, "Успех", f"Выбрана область: {roi.boundingRect()}")
        else:
            QMessageBox.information(self, "Отмена", "Выбор области отменён.")

        # Возобновляем обновление UI от камеры
        thread.pause_updates = False        

    def check_server_status(self):
        # Если серверный поток уже запущен, не запускаем новый
        if hasattr(self, "server_thread") and self.server_thread.isRunning():
            return
        self.server_thread = ServerCheckThread()
        self.server_thread.result_signal.connect(self.handle_server_check_result)
        self.server_thread.start()

    def handle_server_check_result(self, is_active):
        self.server_status_label.setText("Онлайн" if is_active else "Офлайн")

    def load_error_logs(self):
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM error_logs ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            self.error_table.setRowCount(len(rows))
            for row_idx, row in enumerate(rows):
                for col_idx, data in enumerate(row):
                    item = QTableWidgetItem(str(data))
                    if col_idx == 5:
                        item.setToolTip(str(data))
                    self.error_table.setItem(row_idx, col_idx, item)
            conn.close()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки логов: {e}")

    def clear_error_logs(self):
        reply = QMessageBox.question(self, "Подтверждение", "Очистить все логи ошибок?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                conn = sqlite3.connect(DATABASE_FILE)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM error_logs")
                conn.commit()
                conn.close()
                self.load_error_logs()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка очистки логов: {e}")

    def add_video(self):
        from PyQt6.QtWidgets import QFileDialog
        video_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if video_path:
            import os
            video_name = os.path.basename(video_path)
            self.video_name_label.setText(f"Видео: {video_name}")
            base_filename = os.path.splitext(video_name)[0]
            self.video_start = parse_video_filename(base_filename)
            self.video_session_start = time.time()
            if hasattr(self, 'video_thread') and self.video_thread.isRunning():
                self.video_thread.stop()
            self.video_thread = VideoProcessingThread(
                video_path=video_path,
                confidence_threshold=self.confidence_threshold,
                roi=self.detection_roi,
                frame_interval=self.frame_interval,
                parent=self
            )
            self.video_thread.frame_processed.connect(self.update_video_display)
            self.video_thread.detection_signal.connect(self.handle_video_detection)
            self.video_thread.recording_status_changed.connect(self.update_video_recording_status)
            self.video_thread.start()
            self.video_toggle_button.setChecked(True)
            self.video_toggle_button.setIcon(QIcon(STOP_ICON_PATH))
        else:
            print("Видео не выбрано.")

    def on_video_selection_changed(self, index):
        if index > 0 and (index - 1) < len(self.video_list):
            self.play_selected_video(index - 1)

    def play_selected_video(self, video_index):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        video_info = self.video_list[video_index]
        video_path = video_info["path"]
        self.video_name_label.setText(f"Видео: {video_info['name']}")
        self.video_thread = VideoProcessingThread(
            video_path=video_path,
            confidence_threshold=self.confidence_threshold,
            roi=self.detection_roi,
            frame_interval=self.frame_interval,
            parent=self
        )
        self.video_thread.frame_processed.connect(self.update_video_display)
        self.video_thread.detection_signal.connect(self.handle_video_detection)
        self.video_thread.recording_status_changed.connect(self.update_video_recording_status)
        self.video_thread.start()
        self.video_toggle_button.setChecked(True)
        self.video_toggle_button.setIcon(QIcon(STOP_ICON_PATH))

    def toggle_video(self):
        if hasattr(self, 'video_thread'):
            if self.video_toggle_button.isChecked():
                self.video_toggle_button.setIcon(QIcon(STOP_ICON_PATH))
                self.video_thread.resume()
            else:
                self.video_toggle_button.setIcon(QIcon(START_ICON_PATH))
                self.video_thread.pause()

    def select_video_roi(self):
        # Проверяем, есть ли активный поток видео
        current_frame = None
        current_thread = None
        
        if hasattr(self, 'video_thread') and hasattr(self.video_thread, 'last_frame') and self.video_thread.last_frame is not None:
            current_frame = self.video_thread.last_frame
            current_thread = self.video_thread
        
        if current_frame is not None and current_thread is not None:
            roi_dialog = ROISelectorDialog(frame=current_frame, parent=self)
            if roi_dialog.exec() == QDialog.DialogCode.Accepted:
                roi = roi_dialog.get_roi()
                if roi is not None:
                    # Присваиваем выбранное ROI потоку
                    current_thread.roi = roi
                    # Вызываем метод обновления кэша ROI
                    current_thread.update_cached_roi()
                    QMessageBox.information(self, "Успех", "Выбранная область сохранена.")
            else:
                QMessageBox.warning(self, "Ошибка", "Выбор ROI отменен.")
        else:
            QMessageBox.warning(self, "Ошибка", "Нет доступного кадра для выбора ROI. Загрузите видео.")

    def on_video_roi_selected(self, roi):
        self.detection_roi = roi
        QMessageBox.information(self, "Успех", "Выбранная область сохранена.")

    def update_video_display(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.video_display_label.setPixmap(pixmap.scaled(self.video_display_label.size(),
                                                          Qt.AspectRatioMode.KeepAspectRatio,
                                                          Qt.TransformationMode.SmoothTransformation))

    def handle_video_detection(self, detected_count):
        try:
            max_people = int(self.video_max_people_input.text())
        except ValueError:
            max_people = 0
        self.video_count_label.setText(f"Людей: {detected_count}/{max_people}")
        current_time = time.time()
        if max_people > 0 and detected_count > max_people:
            self.video_count_label.setStyleSheet("color: red;")
        else:
            self.video_count_label.setStyleSheet("color: green;")
        if (current_time - self.last_periodic_log_time) >= 40:
            self.last_periodic_log_time = current_time
            from datetime import timedelta
            elapsed_seconds = current_time - self.video_session_start
            event_time = self.video_start + timedelta(seconds=elapsed_seconds)
            event_time_str = format_datetime(event_time)
            video_name = self.video_name_label.text().replace("Видео: ", "")
            # Здесь можно добавить вызов функции для записи статистики видео (например, log_video_event)
            print(f"[DEBUG] Периодическая запись: {event_time_str}, Людей={detected_count}")

    def delete_video(self):
        # Останавливаем все потоки
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
        
        # Очищаем интерфейс
        self.video_display_label.clear()
        self.video_display_label.setText("Видео/Изображение не выбрано")
        self.video_name_label.setText("Видео: не выбрано")
        self.video_count_label.setText("Людей: 0")
        self.video_toggle_button.setChecked(False)
        self.video_toggle_button.setIcon(QIcon(START_ICON_PATH))
        self.video_toggle_button.setEnabled(True)  # Включаем обратно

    def open_add_camera_dialog(self):
        dialog = AddCameraDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            camera_data = dialog.get_camera_data()
            # Проверка на существование камеры по IP
            if not any(cam['ip'] == camera_data['ip'] for cam in self.cameras.values()):
                # Сохраняем камеру в базу данных (если save_cameras реализована, вызовите её здесь)
                from database import save_cameras
                save_cameras([camera_data])
                # После сохранения обновляем список камер из базы
                cameras_list = load_cameras()
                self.cameras = { camera['id']: camera for camera in cameras_list }
                # Ищем ID новой камеры по IP
                new_id = None
                for cam_id, cam in self.cameras.items():
                    if cam['ip'] == camera_data['ip']:
                        new_id = cam_id
                        # Обновляем camera_data, чтобы использовать данные из БД
                        camera_data = cam
                        break
                if new_id is None:
                    QMessageBox.warning(self, "Ошибка", "Не удалось определить ID новой камеры.")
                else:
                    self.add_camera_to_list(new_id, camera_data)
            else:
                QMessageBox.warning(self, "Ошибка", "Камера с таким IP уже добавлена.")

    def add_camera_to_list(self, camera_id, camera_data):
        # Добавляем камеру в область с кнопками (это уже реализовано)
        layout = QHBoxLayout()
        edit_button = QPushButton(camera_data['name'])
        toggle_button = QPushButton()
        toggle_button.setCheckable(True)
        toggle_button.setIcon(QIcon(START_ICON_PATH))
        toggle_button.setStyleSheet("background-color: lightgray;")
        remove_button = QPushButton("Удалить")
        people_count_label = QLabel("Людей: 0")
        max_people_label = QLabel(f"Макс. людей: {camera_data['max_people_count']}")
        edit_button.clicked.connect(lambda: self.open_camera_details_dialog(camera_id, camera_data, edit_button, max_people_label))
        toggle_button.clicked.connect(lambda: self.toggle_camera_with_icons(camera_id, camera_data, people_count_label, toggle_button))
        remove_button.clicked.connect(lambda: self.remove_camera(camera_id))
        layout.addWidget(edit_button)
        layout.addWidget(toggle_button)
        layout.addWidget(remove_button)
        layout.addWidget(people_count_label)
        layout.addWidget(max_people_label)
        self.camera_layout.addLayout(layout)
        
        # Обновляем список камер для выбора (ComboBox)
        self.camera_select_combo.addItem(camera_data['name'])
        self.camera_ui_mapping.append(camera_id)

    def open_camera_details_dialog(self, camera_id, camera_data, edit_button, max_people_label):
        dialog = CameraDetailsDialog(camera_data, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated_data = dialog.get_updated_camera_data()
            try:
                conn = sqlite3.connect(DATABASE_FILE)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE cameras
                    SET name = ?, ip = ?, port = ?, username = ?, password = ?, protocol = ?, path = ?, max_people_count = ?
                    WHERE id = ?
                """, (
                    updated_data['name'], updated_data['ip'], updated_data['port'], updated_data['username'],
                    updated_data['password'], updated_data['protocol'], updated_data['path'],
                    updated_data['max_people_count'], camera_id
                ))
                conn.commit()
                conn.close()
                self.cameras[camera_id] = updated_data
                edit_button.setText(updated_data['name'])
                max_people_label.setText(f"Макс. людей: {updated_data['max_people_count']}")
                if camera_id in self.active_threads:
                    self.active_threads[camera_id].max_people_count = updated_data['max_people_count']
                QMessageBox.information(self, "Успех", "Параметры камеры успешно обновлены!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при обновлении базы данных: {e}")

    def toggle_camera_with_icons(self, camera_id, camera_data, people_count_label, toggle_button):
        if toggle_button.isChecked():
            toggle_button.setIcon(QIcon(r"C:\Users\Developer1\Downloads\pause_player.svg"))
            toggle_button.setStyleSheet("background-color: lightgreen;")
            self.start_camera(camera_id, camera_data, people_count_label)
        else:
            toggle_button.setIcon(QIcon(START_ICON_PATH))
            toggle_button.setStyleSheet("background-color: lightgray;")
            self.stop_camera(camera_id)

    def start_camera(self, camera_id, camera_data, people_count_label):
        # Если поток уже существует, сначала останавливаем его
        if camera_id in self.active_threads:
            old_thread = self.active_threads[camera_id]
            if old_thread.isRunning():
                old_thread.stop()
                old_thread.wait(3000)  # Ждем завершения старого потока
            del self.active_threads[camera_id]
            
            # Небольшая задержка для освобождения ресурсов камеры
            import time
            time.sleep(0.5)
        
        # Создаем новый поток
        camera_data['hospital_name'] = self.hospital_name_input.text() or "Не указано"
        thread = CameraThread(camera_id, camera_data, self.frame_interval)
        thread.frame_signal.connect(lambda count, cam_id=camera_id: self.update_people_count(count, people_count_label, cam_id))
        thread.frame_display_signal.connect(lambda cam_id, img: self.update_frame_display(cam_id, img))
        thread.recording_status_changed.connect(self.update_camera_recording_status)
        self.active_threads[camera_id] = thread
        thread.start()
        
        # Если камера ещё не присутствует в маппинге, добавляем её.
        if camera_id not in self.camera_ui_mapping:
            self.camera_ui_mapping.append(camera_id)



    def update_people_count(self, count, label, cam_id):
        label.setText(f"Людей: {count}")

    def update_frame_display(self, cam_id, img):
        if self.current_camera == cam_id:
            pixmap = QPixmap.fromImage(img)
            self.stream_screen.setPixmap(pixmap.scaled(self.stream_screen.size(),
                                                        Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))

    def select_camera_for_stream(self, index):
        # Первый элемент ComboBox – заглушка ("Выберите камеру для трансляции")
        if index > 0 and index - 1 < len(self.camera_ui_mapping):
            cam_id = self.camera_ui_mapping[index - 1]
            self.current_camera = cam_id
            print("Выбрана камера с ID:", cam_id)
        else:
            self.current_camera = None
            print("Камера не выбрана")
    def remove_camera(self, camera_id):
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            conn.commit()
            conn.close()
            del self.cameras[camera_id]
            self.reload_camera_list()
        except Exception as e:
            print(f"Ошибка при удалении камеры: {e}")


    def stop_camera(self, camera_id):
        if camera_id in self.active_threads:
            thread = self.active_threads[camera_id]
            thread.stop()  # Останавливаем поток камеры
            thread.wait(3000)  # Ждем завершения потока
            del self.active_threads[camera_id]

    def cleanup_thread(self, camera_id):
        if camera_id in self.active_threads:
            del self.active_threads[camera_id]
        # Дополнительно можно обновить UI, удалив камеру из ComboBox или списка камер        
               

    def reload_camera_list(self):
        for i in reversed(range(self.camera_layout.count())):
            item = self.camera_layout.itemAt(i)
            if item is not None and item.widget():
                item.widget().deleteLater()
        self.load_saved_cameras()

    def load_saved_cameras(self):
        for camera_id, camera_data in self.cameras.items():
            self.add_camera_to_list(camera_id, camera_data)

    def handle_error(self, error_msg, camera_id, stack_trace):
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO error_logs (error_message, camera_id, camera_name, stack_trace)
                VALUES (?, ?, ?, ?)
            """, (error_msg, camera_id, self.cameras.get(camera_id, {}).get('name', 'Неизвестная камера'), stack_trace))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка записи в логи: {e}")
        QMessageBox.critical(self, "Ошибка", f"Ошибка в камере {camera_id}:\n{error_msg}")


    def save_settings(self):
        """
        Сохраняет настройки в таблицу settings.
        """
        try:
            import sqlite3
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            # Удаляем старые настройки
            cursor.execute("DELETE FROM settings")
            # Вставляем новые настройки
            cursor.execute("""
                INSERT INTO settings (frame_interval, hospital_name, confidence_threshold, 
                                    enable_face_recognition, enable_person_recognition, 
                                    enable_face_person_linking, enable_tracking_history, 
                                    enable_persistent_storage, face_confidence_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.frame_interval,
                self.hospital_name_input.text(),
                int(self.confidence_threshold * 100),  # сохраняем в процентах
                self.cb_face_recognition.isChecked(),
                self.cb_person_recognition.isChecked(),
                self.cb_face_person_linking.isChecked(),
                self.cb_tracking_history.isChecked(),
                self.cb_persistent_storage.isChecked(),
                int(getattr(self, 'face_confidence_threshold', FACE_THRESHOLD) * 100)  # сохраняем в процентах
            ))
            conn.commit()
            conn.close()
            
            # Сохраняем настройку в config.py
            import config
            config.FACE_THRESHOLD = getattr(self, 'face_confidence_threshold', FACE_THRESHOLD)
            
            QMessageBox.information(self, "Успех", "Настройки успешно сохранены!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении настроек: {e}")

    def load_settings(self):
        """
        Загружает настройки из таблицы settings и обновляет элементы интерфейса.
        """
        try:
            import sqlite3
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT frame_interval, hospital_name, confidence_threshold, 
                       enable_face_recognition, enable_person_recognition, 
                       enable_face_person_linking, enable_tracking_history, 
                       enable_persistent_storage, face_confidence_threshold
                FROM settings LIMIT 1
            """)
            settings = cursor.fetchone()
            if settings:
                self.frame_interval = settings[0] if settings[0] is not None else DEFAULT_FRAME_INTERVAL
                self.hospital_name_input.setText(settings[1])
                self.confidence_input.setText(str(settings[2]))
                self.confidence_label.setText(f"Уверенность: {settings[2]}%")
                self.confidence_threshold = settings[2] / 100.0
                
                # Загружаем настройки флагов
                if len(settings) > 3:
                    self.cb_face_recognition.setChecked(bool(settings[3]))
                    self.cb_person_recognition.setChecked(bool(settings[4]))
                    self.cb_face_person_linking.setChecked(bool(settings[5]))
                    self.cb_tracking_history.setChecked(bool(settings[6]))
                    self.cb_persistent_storage.setChecked(bool(settings[7]))
                    
                    # Обновляем конфигурацию
                    config.ENABLE_FACE_RECOGNITION = bool(settings[3])
                    config.ENABLE_PERSON_RECOGNITION = bool(settings[4])
                    config.ENABLE_FACE_PERSON_LINKING = bool(settings[5])
                    config.ENABLE_TRACKING_HISTORY = bool(settings[6])
                    config.ENABLE_PERSISTENT_STORAGE = bool(settings[7])
                
                # Загружаем настройку уверенности детекции лиц
                if len(settings) > 8 and settings[8] is not None:
                    face_confidence = settings[8]
                    self.face_confidence_threshold = face_confidence / 100.0
                    if hasattr(self, 'face_confidence_input'):
                        self.face_confidence_input.setText(str(face_confidence))
                        self.face_confidence_label.setText(f"Уверенность: {face_confidence}%")
                    config.FACE_THRESHOLD = self.face_confidence_threshold
                else:
                    # Значение по умолчанию
                    self.face_confidence_threshold = FACE_THRESHOLD
                    if hasattr(self, 'face_confidence_input'):
                        self.face_confidence_input.setText(str(int(FACE_THRESHOLD * 100)))
                        self.face_confidence_label.setText(f"Уверенность: {int(FACE_THRESHOLD * 100)}%")
                
                # Обновляем также выпадающий список частоты кадров, если он существует
                if self.frame_interval_combo:
                    self.frame_interval_combo.setCurrentText(str(self.frame_interval))
            else:
                # Если настроек нет, задаем значения по умолчанию
                self.frame_interval = DEFAULT_FRAME_INTERVAL
                self.face_confidence_threshold = FACE_THRESHOLD
            conn.close()
        except Exception as e:
            print(f"Ошибка при загрузке настроек из базы данных: {e}")

    def update_face_stats(self):
        """
        Обновляет статистику уникальных лиц в статус баре.
        """
        try:
            count = get_faces_count()
            self.statusBar().showMessage(f"Уникальных лиц в базе: {count}")
        except Exception as e:
            print(f"Ошибка при обновлении статистики лиц: {e}")

    def open_set_face_name_dialog(self):
        dialog = SetFaceNameDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            face_id = dialog.get_face_id()
            name = dialog.get_name()
            if face_id is not None and name:
                try:
                    set_face_name(face_id, name)
                    QMessageBox.information(self, "Успех", f"Имя '{name}' установлено для лица ID: {face_id}")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка: {str(e)}")
            else:
                QMessageBox.warning(self, "Ошибка", "Введите корректный Face ID и имя")

    def update_face_recognition_setting(self, state):
        """Обновляет настройку распознавания лиц в реальном времени"""
        config.ENABLE_FACE_RECOGNITION = bool(state)
        self.update_active_threads_settings()
        update_settings_flags()  # Обновляем настройки в базе данных
        print(f"Распознавание лиц {'включено' if state else 'отключено'}")

    def update_person_recognition_setting(self, state):
        """Обновляет настройку распознавания людей в реальном времени"""
        config.ENABLE_PERSON_RECOGNITION = bool(state)
        self.update_active_threads_settings()
        update_settings_flags()  # Обновляем настройки в базе данных
        print(f"Распознавание людей {'включено' if state else 'отключено'}")

    def update_face_person_linking_setting(self, state):
        """Обновляет настройку связывания лиц и людей в реальном времени"""
        config.ENABLE_FACE_PERSON_LINKING = bool(state)
        self.update_active_threads_settings()
        update_settings_flags()  # Обновляем настройки в базе данных
        print(f"Связывание лиц и людей {'включено' if state else 'отключено'}")

    def update_tracking_history_setting(self, state):
        """Обновляет настройку сохранения истории треков в реальном времени"""
        config.ENABLE_TRACKING_HISTORY = bool(state)
        self.update_active_threads_settings()
        update_settings_flags()  # Обновляем настройки в базе данных
        print(f"Сохранение истории треков {'включено' if state else 'отключено'}")

    def update_persistent_storage_setting(self, state):
        """Обновляет настройку сохранения эмбеддингов в реальном времени"""
        config.ENABLE_PERSISTENT_STORAGE = bool(state)
        self.update_active_threads_settings()
        update_settings_flags()  # Обновляем настройки в базе данных
        print(f"Сохранение эмбеддингов {'включено' if state else 'отключено'}")

    def update_active_threads_settings(self):
        """Обновляет настройки во всех активных потоках"""
        # Обновляем настройки в базе данных
        update_settings_flags()
        
        # Обновляем настройки во всех активных потоках
        for thread in self.active_threads.values():
            if hasattr(thread, 'update_settings'):
                thread.update_settings()



    def open_manual_merge_faces_dialog(self):
        """Открывает диалог для ручного объединения лиц по имени"""
        dialog = ManualMergeFacesDialog(self)
        dialog.exec()

    def open_auto_merge_faces_dialog(self):
        """Открывает диалог для автоматического объединения лиц по имени"""
        dialog = AutoMergeFacesDialog(self)
        dialog.exec()

    def open_photobot_search_dialog(self):
        """Открывает диалог для поиска совпадений фоторобота"""
        dialog = PhotobotSearchDialog(self)
        dialog.exec()

    def toggle_camera_recording(self):
        """
        Переключает запись видео для активной камеры.
        """
        try:
            # Находим активную камеру
            active_camera_id = None
            for camera_id, thread in self.active_threads.items():
                if isinstance(thread, CameraThread) and thread.isRunning():
                    active_camera_id = camera_id
                    break
            
            if active_camera_id is None:
                QMessageBox.warning(self, "Предупреждение", "Нет активной камеры для записи")
                self.camera_recording_button.setChecked(False)
                return
            
            thread = self.active_threads[active_camera_id]
            
            if self.camera_recording_button.isChecked():
                # Начинаем запись
                if thread.start_video_recording():
                    self.camera_recording_button.setText("Остановить запись видео")
                    self.camera_recording_status_label.setText("Статус записи: Записывается")
                    self.camera_recording_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                else:
                    self.camera_recording_button.setChecked(False)
                    QMessageBox.warning(self, "Ошибка", "Не удалось начать запись видео")
            else:
                # Останавливаем запись
                if thread.stop_video_recording():
                    self.camera_recording_button.setText("Начать запись видео")
                    self.camera_recording_status_label.setText("Статус записи: Остановлена")
                    self.camera_recording_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                else:
                    self.camera_recording_button.setChecked(True)
                    QMessageBox.warning(self, "Ошибка", "Не удалось остановить запись видео")
                    
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при управлении записью видео: {str(e)}")
            self.camera_recording_button.setChecked(False)
    
    def toggle_video_recording(self):
        """
        Переключает запись видео для активного видеофайла.
        """
        try:
            if not hasattr(self, 'video_thread') or self.video_thread is None:
                QMessageBox.warning(self, "Предупреждение", "Нет активного видео для записи")
                self.video_recording_button.setChecked(False)
                return
            
            if self.video_recording_button.isChecked():
                # Начинаем запись
                if self.video_thread.start_video_recording():
                    self.video_recording_button.setText("Остановить запись видео")
                    self.video_recording_status_label.setText("Статус записи: Записывается")
                    self.video_recording_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                else:
                    self.video_recording_button.setChecked(False)
                    QMessageBox.warning(self, "Ошибка", "Не удалось начать запись видео")
            else:
                # Останавливаем запись
                if self.video_thread.stop_video_recording():
                    self.video_recording_button.setText("Начать запись видео")
                    self.video_recording_status_label.setText("Статус записи: Остановлена")
                    self.video_recording_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                else:
                    self.video_recording_button.setChecked(True)
                    QMessageBox.warning(self, "Ошибка", "Не удалось остановить запись видео")
                    
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при управлении записью видео: {str(e)}")
            self.video_recording_button.setChecked(False)
    
    def update_camera_recording_status(self, camera_id, is_recording):
        """
        Обновляет статус записи для камеры.
        """
        try:
            if is_recording:
                self.camera_recording_button.setText("Остановить запись видео")
                self.camera_recording_status_label.setText("Статус записи: Записывается")
                self.camera_recording_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            else:
                self.camera_recording_button.setText("Начать запись видео")
                self.camera_recording_status_label.setText("Статус записи: Остановлена")
                self.camera_recording_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.camera_recording_button.setChecked(False)
        except Exception as e:
            print(f"Ошибка обновления статуса записи камеры: {e}")
    
    def update_video_recording_status(self, is_recording):
        """
        Обновляет статус записи для видеофайла.
        """
        try:
            if is_recording:
                self.video_recording_button.setText("Остановить запись видео")
                self.video_recording_status_label.setText("Статус записи: Записывается")
                self.video_recording_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            else:
                self.video_recording_button.setText("Начать запись видео")
                self.video_recording_status_label.setText("Статус записи: Остановлена")
                self.video_recording_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.video_recording_button.setChecked(False)
        except Exception as e:
            print(f"Ошибка обновления статуса записи видео: {e}")

    def open_photobot_history_dialog(self):
        """Открывает диалог для просмотра истории поиска фотороботов"""
        dialog = PhotobotHistoryDialog(self)
        dialog.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

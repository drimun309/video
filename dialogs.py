# dialogs.py
import cv2
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QImage, QPixmap, QPolygon, QColor
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QScrollArea, QHBoxLayout, QTableWidget, QLineEdit
from PyQt6.QtWidgets import QApplication # Added for QApplication.processEvents()

import sys
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QTableWidget, QTableWidgetItem, QMessageBox,
                             QComboBox, QFormLayout, QTextEdit, QScrollArea, QWidget)
from PyQt6.QtCore import Qt
from database import get_faces_with_names, get_faces_by_name, merge_faces_by_name_with_embedding, auto_merge_faces_by_names


class PolygonSelectionLabel(QLabel):
    polygonSelected = pyqtSignal(QPolygon)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []        # Список точек многоугольника
        self.drawing = True     # Флаг: если True – добавляем точки, если False – уже сформирован, можно перемещать
        self.dragging = False   # Флаг перетаскивания
        self.last_drag_position = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pt = event.position().toPoint()
            # Если многоугольник уже сформирован и клик внутри него — начинаем перетаскивание
            if not self.drawing and QPolygon(self.points).containsPoint(pt, Qt.FillRule.OddEvenFill):
                self.dragging = True
                self.last_drag_position = pt
                print("[DEBUG] Начало перетаскивания ROI.")
            else:
                # Иначе добавляем точку
                self.points.append(pt)
                print("[DEBUG] Добавлена точка:", pt)
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            # При правом клике завершаем выбор, если точек достаточно
            if len(self.points) >= 3:
                
                self.drawing = False
                polygon = QPolygon(self.points)
                print("[DEBUG] Завершение выбора многоугольника. Точки:", self.points)
                print("[DEBUG] Огибающий прямоугольник:", polygon.boundingRect())
                self.polygonSelected.emit(polygon)
            else:
                print("[DEBUG] Недостаточно точек для формирования многоугольника.")
            self.update()

    def mouseMoveEvent(self, event):
        if self.dragging:
            new_pos = event.position().toPoint()
            offset = new_pos - self.last_drag_position
            self.points = [pt + offset for pt in self.points]
            self.last_drag_position = new_pos
            print("[DEBUG] Перемещаем ROI. Смещение:", offset)
            self.update()
        elif self.drawing:
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            self.dragging = False
            print("[DEBUG] Завершено перетаскивание ROI.")
            self.polygonSelected.emit(QPolygon(self.points))
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.points:
            painter = QPainter(self)
            pen = QPen(Qt.GlobalColor.red, 2)
            painter.setPen(pen)
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i], self.points[i+1])
            if self.drawing or self.dragging:
                # Рисуем линию от последней точки до текущей позиции курсора
                current_pos = self.mapFromGlobal(self.cursor().pos())
                painter.drawLine(self.points[-1], current_pos)

class ROISelectorDialog(QDialog):
    roiChosen = pyqtSignal(object)

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выделите произвольную область для детекции")
        self.resize(800, 600)
        layout = QVBoxLayout(self)

        # Преобразуем кадр из BGR в RGB и создаём QImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        self.q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Создаем QPixmap в оригинальном размере
        pixmap = QPixmap.fromImage(self.q_img)

        # Используем наш виджет для произвольного выделения
        self.image_label = PolygonSelectionLabel()
        self.image_label.setPixmap(pixmap)

        # Оборачиваем изображение в QScrollArea, чтобы можно было прокручивать, если изображение больше окна
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(False)
        layout.addWidget(scroll_area)

        # Кнопка подтверждения выбора
        confirm_btn = QPushButton("Подтвердить")
        confirm_btn.clicked.connect(self.accept)
        layout.addWidget(confirm_btn)

        self.selected_polygon = None
        self.image_label.polygonSelected.connect(self.on_polygon_selected)

        # Поскольку изображение не масштабируется, отображаемый размер совпадает с оригиналом
        self.original_size = pixmap.size()
        self.displayed_size = pixmap.size()

    def on_polygon_selected(self, polygon):
        print("[DEBUG] Получен многоугольник:", polygon)
        print("[DEBUG] Его огибающий прямоугольник:", polygon.boundingRect())
        self.selected_polygon = polygon
        self.roiChosen.emit(polygon)

    def get_roi(self):
        return self.selected_polygon


class SetFaceNameDialog(QDialog):
    """
    Диалог для установки имени лицу
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Задать имя лицу")
        self.setModal(True)
        self.resize(300, 150)
        
        layout = QFormLayout(self)
        
        self.face_id_input = QLineEdit()
        self.face_id_input.setPlaceholderText("Введите ID лица")
        layout.addRow("ID лица:", self.face_id_input)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Введите имя")
        layout.addRow("Имя:", self.name_input)
        
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Сохранить")
        self.cancel_button = QPushButton("Отмена")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addRow(button_layout)
    
    def get_face_id(self):
        try:
            return int(self.face_id_input.text())
        except ValueError:
            return None
    
    def get_name(self):
        return self.name_input.text().strip()

class ManualMergeFacesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ручное объединение лиц")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Объяснение
        explanation = QLabel("Выберите имя для объединения всех лиц с этим именем:")
        layout.addWidget(explanation)
        
        # Выбор имени
        self.name_combo = QComboBox()
        self.name_combo.currentTextChanged.connect(self.update_faces_table)
        layout.addWidget(self.name_combo)
        
        # Таблица лиц
        self.faces_table = QTableWidget()
        self.faces_table.setColumnCount(4)
        self.faces_table.setHorizontalHeaderLabels(["ID", "Имя", "Первое появление", "Последнее появление"])
        layout.addWidget(self.faces_table)
        
        # Кнопки
        button_layout = QHBoxLayout()
        self.merge_button = QPushButton("Объединить")
        self.cancel_button = QPushButton("Отмена")
        
        self.merge_button.clicked.connect(self.merge_faces)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.merge_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.load_names()
    
    def load_names(self):
        """Загружает список уникальных имён"""
        faces = get_faces_with_names()
        names = list(set([face[1] for face in faces]))
        names.sort()
        
        self.name_combo.clear()
        self.name_combo.addItem("Выберите имя")
        self.name_combo.addItems(names)
    
    def update_faces_table(self, name):
        """Обновляет таблицу лиц для выбранного имени"""
        if name == "Выберите имя" or not name:
            self.faces_table.setRowCount(0)
            return
        
        faces = get_faces_by_name(name)
        self.faces_table.setRowCount(len(faces))
        
        for row, face in enumerate(faces):
            face_id, embedding, first_seen, last_seen = face
            self.faces_table.setItem(row, 0, QTableWidgetItem(str(face_id)))
            self.faces_table.setItem(row, 1, QTableWidgetItem(name))
            self.faces_table.setItem(row, 2, QTableWidgetItem(str(first_seen)))
            self.faces_table.setItem(row, 3, QTableWidgetItem(str(last_seen)))
    
    def merge_faces(self):
        """Объединяет лица с выбранным именем"""
        name = self.name_combo.currentText()
        if name == "Выберите имя" or not name:
            QMessageBox.warning(self, "Ошибка", "Выберите имя для объединения")
            return
        
        faces = get_faces_by_name(name)
        if len(faces) < 2:
            QMessageBox.information(self, "Информация", f"Для имени '{name}' найдено менее 2 лиц. Объединение не требуется.")
            return
        
        reply = QMessageBox.question(
            self, "Подтверждение", 
            f"Объединить {len(faces)} лиц с именем '{name}'? Это действие нельзя отменить.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if merge_faces_by_name_with_embedding(name):
                QMessageBox.information(self, "Успех", f"Успешно объединено {len(faces)} лиц с именем '{name}'")
                self.load_names()  # Обновляем список имён
                self.update_faces_table(name)  # Обновляем таблицу
            else:
                QMessageBox.critical(self, "Ошибка", "Произошла ошибка при объединении лиц")

class AutoMergeFacesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Автоматическое объединение лиц")
        self.setModal(True)
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Объяснение
        explanation = QLabel("Автоматически объединит все лица с одинаковыми именами:")
        layout.addWidget(explanation)
        
        # Кнопка запуска
        self.start_button = QPushButton("Начать автоматическое объединение")
        self.start_button.clicked.connect(self.start_auto_merge)
        layout.addWidget(self.start_button)
        
        # Область результатов
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Результаты объединения появятся здесь...")
        layout.addWidget(self.results_text)
        
        # Кнопка закрытия
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)
    
    def start_auto_merge(self):
        """Запускает автоматическое объединение"""
        self.start_button.setEnabled(False)
        self.results_text.clear()
        self.results_text.append("Начинаем автоматическое объединение...\n")
        
        try:
            results = auto_merge_faces_by_names()
            
            if not results:
                self.results_text.append("Нет лиц для объединения.")
            else:
                self.results_text.append(f"Объединено {len(results)} групп лиц:\n")
                for name, count in results.items():
                    self.results_text.append(f"• '{name}': объединено {count} лиц")
                
                total_merged = sum(results.values())
                self.results_text.append(f"\nВсего объединено лиц: {total_merged}")
            
        except Exception as e:
            self.results_text.append(f"Ошибка при объединении: {str(e)}")
        
        finally:
            self.start_button.setEnabled(True)


# === Диалог для работы с фотороботами ===

class PhotobotSearchDialog(QDialog):
    """
    Диалог для загрузки фоторобота и поиска совпадений в базе лиц
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Поиск совпадений для фоторобота")
        self.setModal(True)
        self.resize(800, 600)
        
        # Импорты для обработки изображений
        import cv2
        import numpy as np
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        from config import FACE_THRESHOLD
        
        self.cv2 = cv2
        self.np = np
        self.torch = torch
        self.FACE_THRESHOLD = FACE_THRESHOLD
        
        # Инициализация моделей
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.photobot_image = None
        self.photobot_embedding = None
        self.search_results = []
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Поиск совпадений для фоторобота")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Секция загрузки изображения
        load_group = QWidget()
        load_layout = QHBoxLayout(load_group)
        
        self.load_button = QPushButton("Загрузить фоторобот")
        self.load_button.clicked.connect(self.load_photobot)
        load_layout.addWidget(self.load_button)
        
        self.image_label = QLabel("Изображение не загружено")
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #ccc;")
        load_layout.addWidget(self.image_label)
        
        layout.addWidget(load_group)
        
        # Секция настроек поиска
        settings_group = QWidget()
        settings_layout = QFormLayout(settings_group)
        
        self.photobot_id_input = QLineEdit()
        self.photobot_id_input.setPlaceholderText("Введите ID фоторобота")
        settings_layout.addRow("ID фоторобота:", self.photobot_id_input)
        
        self.threshold_input = QLineEdit("0.66")
        self.threshold_input.setPlaceholderText("Порог совпадения (0.0-1.0)")
        settings_layout.addRow("Порог совпадения:", self.threshold_input)
        
        self.top_k_input = QLineEdit("5")
        self.top_k_input.setPlaceholderText("Количество лучших совпадений")
        settings_layout.addRow("Топ совпадений:", self.top_k_input)
        
        layout.addWidget(settings_group)
        
        # Кнопка поиска
        self.search_button = QPushButton("Найти совпадения")
        self.search_button.clicked.connect(self.search_matches)
        self.search_button.setEnabled(False)
        layout.addWidget(self.search_button)
        
        # Результаты поиска
        results_label = QLabel("Результаты поиска:")
        results_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Место", "Face ID", "Имя", "Уверенность", "Последнее появление"
        ])
        layout.addWidget(self.results_table)
        
        # Кнопки действий
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Сохранить результат")
        self.save_button.clicked.connect(self.save_search_result)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def load_photobot(self):
        """Загружает изображение фоторобота"""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение фоторобота",
            "",
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            try:
                # Загружаем изображение
                image = self.cv2.imread(file_path)
                if image is None:
                    QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                    return
                
                # Конвертируем в RGB
                image_rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
                
                # Отображаем изображение
                h, w, ch = image_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # Масштабируем для отображения
                scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
                self.image_label.setPixmap(scaled_pixmap)
                
                # Сохраняем изображение для обработки
                self.photobot_image = image_rgb
                
                # Извлекаем эмбеддинг
                self.extract_embedding()
                
                self.search_button.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке изображения: {str(e)}")
    
    def extract_embedding(self):
        """Извлекает эмбеддинг лица из изображения фоторобота"""
        try:
            if self.photobot_image is None:
                return
            
            # Детектируем лицо
            boxes, _ = self.mtcnn.detect(self.photobot_image)
            
            if boxes is None or len(boxes) == 0:
                QMessageBox.warning(self, "Предупреждение", "Лицо не обнаружено на изображении")
                return
            
            # Извлекаем эмбеддинг (исправленный вызов extract)
            face_tensors = self.mtcnn.extract(self.photobot_image, boxes, save_path=None)
            if face_tensors is None or len(face_tensors) == 0:
                QMessageBox.warning(self, "Предупреждение", "Не удалось извлечь признаки лица")
                return
            face_tensor = face_tensors[0]
            # Универсальная проверка и преобразование каналов
            import sys
            print('face_tensor.shape:', face_tensor.shape, file=sys.stderr)
            print('face_tensor.dtype:', face_tensor.dtype, file=sys.stderr)
            if face_tensor.ndim == 2:
                # [H, W] -> [1, H, W]
                face_tensor = face_tensor.unsqueeze(0)
            if face_tensor.shape[0] == 1:
                face_tensor = face_tensor.repeat(3, 1, 1)
            elif face_tensor.shape[0] != 3:
                raise ValueError(f'Unexpected number of channels: {face_tensor.shape[0]}')
            print('face_tensor.shape after fix:', face_tensor.shape, file=sys.stderr)
            face_tensor = face_tensor.to(self.device)  # Явно переносим на GPU/CPU
            embedding = self.resnet(face_tensor.unsqueeze(0))
            embedding_np = embedding.detach().cpu().numpy()
            
            # Сохраняем эмбеддинг
            self.photobot_embedding = embedding_np.tobytes()
            
            QMessageBox.information(self, "Успех", "Эмбеддинг лица успешно извлечен")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при извлечении эмбеддинга: {str(e)}")
    
    def search_matches(self):
        """Ищет совпадения в базе данных"""
        try:
            if self.photobot_embedding is None:
                QMessageBox.warning(self, "Предупреждение", "Сначала загрузите фоторобот")
                return
            
            # Получаем параметры поиска
            threshold = float(self.threshold_input.text())
            top_k = int(self.top_k_input.text())
            
            # Импортируем функцию поиска
            from database import find_top_matches_for_photobot
            
            # Выполняем поиск
            self.search_results = find_top_matches_for_photobot(
                self.photobot_embedding, 
                top_k=top_k, 
                threshold=threshold
            )
            
            # Отображаем результаты
            self.display_results()
            
            self.save_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при поиске совпадений: {str(e)}")
    
    def display_results(self):
        """Отображает результаты поиска в таблице"""
        self.results_table.setRowCount(len(self.search_results))
        
        for i, match in enumerate(self.search_results):
            # Место
            place_item = QTableWidgetItem(str(i + 1))
            self.results_table.setItem(i, 0, place_item)
            
            # Face ID
            face_id_item = QTableWidgetItem(str(match['face_id']))
            self.results_table.setItem(i, 1, face_id_item)
            
            # Имя
            name_item = QTableWidgetItem(match['name'])
            self.results_table.setItem(i, 2, name_item)
            
            # Уверенность
            confidence_item = QTableWidgetItem(f"{match['confidence']:.3f}")
            self.results_table.setItem(i, 3, confidence_item)
            
            # Последнее появление
            last_seen_item = QTableWidgetItem(match['last_seen'])
            self.results_table.setItem(i, 4, last_seen_item)
            
            # Раскрашиваем строку в зависимости от уверенности
            if match['confidence'] >= 0.8:
                color = QColor(144, 238, 144)  # Светло-зеленый для высокого совпадения
            elif match['confidence'] >= 0.6:
                color = QColor(255, 255, 224)  # Светло-желтый для среднего совпадения
            else:
                color = QColor(255, 182, 193)  # Светло-розовый для низкого совпадения
            
            for j in range(5):
                item = self.results_table.item(i, j)
                item.setBackground(color)
                # Устанавливаем темный цвет текста для лучшей читаемости
                item.setForeground(QColor(0, 0, 0))  # Черный текст
        
        self.results_table.resizeColumnsToContents()
    
    def save_search_result(self):
        """Сохраняет результат поиска в базу данных"""
        try:
            photobot_id = self.photobot_id_input.text().strip()
            if not photobot_id:
                QMessageBox.warning(self, "Предупреждение", "Введите ID фоторобота")
                return
            
            if not self.search_results:
                QMessageBox.warning(self, "Предупреждение", "Нет результатов для сохранения")
                return
            
            # Импортируем функцию сохранения
            from database import save_photobot_search_result
            
            # Сохраняем результат
            success = save_photobot_search_result(
                photobot_id=photobot_id,
                embedding_bytes=self.photobot_embedding,
                search_results=self.search_results
            )
            
            if success:
                QMessageBox.information(self, "Успех", "Результат поиска сохранен в базу данных")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить результат")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")


class PhotobotHistoryDialog(QDialog):
    """
    Диалог для просмотра истории поиска фотороботов
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("История поиска фотороботов")
        self.setModal(True)
        self.resize(900, 600)
        
        self.init_ui()
        self.load_history()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("История поиска фотороботов")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Таблица истории
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "ID", "ID фоторобота", "Время поиска", "Лучшее совпадение", 
            "Уверенность", "Дата создания"
        ])
        self.history_table.doubleClicked.connect(self.show_details)
        layout.addWidget(self.history_table)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Обновить")
        self.refresh_button.clicked.connect(self.load_history)
        button_layout.addWidget(self.refresh_button)
        
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def load_history(self):
        """Загружает историю поиска"""
        try:
            from database import get_photobot_search_history
            
            history = get_photobot_search_history(limit=100)
            
            self.history_table.setRowCount(len(history))
            
            for i, record in enumerate(history):
                # ID
                id_item = QTableWidgetItem(str(record['id']))
                self.history_table.setItem(i, 0, id_item)
                
                # ID фоторобота
                photobot_id_item = QTableWidgetItem(record['photobot_id'])
                self.history_table.setItem(i, 1, photobot_id_item)
                
                # Время поиска
                search_time_item = QTableWidgetItem(record['search_timestamp'])
                self.history_table.setItem(i, 2, search_time_item)
                
                # Лучшее совпадение
                match_name_item = QTableWidgetItem(record['top_match_name'] or "Нет совпадений")
                self.history_table.setItem(i, 3, match_name_item)
                
                # Уверенность
                confidence_item = QTableWidgetItem(
                    f"{record['top_match_confidence']:.3f}" if record['top_match_confidence'] else "N/A"
                )
                self.history_table.setItem(i, 4, confidence_item)
                
                # Дата создания
                created_item = QTableWidgetItem(record['created_at'])
                self.history_table.setItem(i, 5, created_item)
                
                # Раскрашиваем строку в зависимости от уверенности
                if record['top_match_confidence']:
                    if record['top_match_confidence'] >= 0.8:
                        color = QColor(144, 238, 144)  # Светло-зеленый
                    elif record['top_match_confidence'] >= 0.6:
                        color = QColor(255, 255, 224)  # Светло-желтый
                    else:
                        color = QColor(255, 182, 193)  # Светло-розовый
                    
                    for j in range(6):
                        item = self.history_table.item(i, j)
                        item.setBackground(color)
                        # Устанавливаем темный цвет текста для лучшей читаемости
                        item.setForeground(QColor(0, 0, 0))  # Черный текст
            
            self.history_table.resizeColumnsToContents()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке истории: {str(e)}")
    
    def show_details(self, index):
        """Показывает детали конкретного поиска"""
        try:
            search_id = int(self.history_table.item(index.row(), 0).text())
            
            from database import get_photobot_search_details
            
            details = get_photobot_search_details(search_id)
            if details:
                self.show_search_details(details)
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить детали")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке деталей: {str(e)}")
    
    def show_search_details(self, details):
        """Показывает детальное окно с результатами поиска"""
        detail_dialog = QDialog(self)
        detail_dialog.setWindowTitle(f"Детали поиска фоторобота {details['photobot_id']}")
        detail_dialog.setModal(True)
        detail_dialog.resize(700, 500)
        
        layout = QVBoxLayout(detail_dialog)
        
        # Информация о поиске
        info_text = f"""
        <h3>Информация о поиске:</h3>
        <p><b>ID фоторобота:</b> {details['photobot_id']}</p>
        <p><b>Время поиска:</b> {details['search_timestamp']}</p>
        <p><b>Дата создания:</b> {details['created_at']}</p>
        """
        
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        # Результаты поиска
        results_label = QLabel("Результаты поиска:")
        results_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)
        
        results_table = QTableWidget()
        results_table.setColumnCount(5)
        results_table.setHorizontalHeaderLabels([
            "Место", "Face ID", "Имя", "Уверенность", "Последнее появление"
        ])
        
        matches = details['all_matches']
        results_table.setRowCount(len(matches))
        
        for i, match in enumerate(matches):
            # Место
            place_item = QTableWidgetItem(str(i + 1))
            results_table.setItem(i, 0, place_item)
            
            # Face ID
            face_id_item = QTableWidgetItem(str(match['face_id']))
            results_table.setItem(i, 1, face_id_item)
            
            # Имя
            name_item = QTableWidgetItem(match['name'])
            results_table.setItem(i, 2, name_item)
            
            # Уверенность
            confidence_item = QTableWidgetItem(f"{match['confidence']:.3f}")
            results_table.setItem(i, 3, confidence_item)
            
            # Последнее появление
            last_seen_item = QTableWidgetItem(match['last_seen'])
            results_table.setItem(i, 4, last_seen_item)
            
            # Раскрашиваем строку
            if match['confidence'] >= 0.8:
                color = QColor(144, 238, 144)  # Светло-зеленый
            elif match['confidence'] >= 0.6:
                color = QColor(255, 255, 224)  # Светло-желтый
            else:
                color = QColor(255, 182, 193)  # Светло-розовый
            
            for j in range(5):
                item = results_table.item(i, j)
                item.setBackground(color)
                # Устанавливаем темный цвет текста для лучшей читаемости
                item.setForeground(QColor(0, 0, 0))  # Черный текст
        
        results_table.resizeColumnsToContents()
        layout.addWidget(results_table)
        
        # Кнопка закрытия
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(detail_dialog.accept)
        layout.addWidget(close_button)
        
        detail_dialog.exec()



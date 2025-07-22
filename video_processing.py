# video_processing.py

# Этот файл теперь служит главным импортом для всех компонентов видеообработки
# Все классы и функции разделены на логические модули для лучшей организации кода

# Импортируем все основные компоненты из разделенных модулей
from detection_processing import (
    # Модели и глобальные объекты
    model, tracker, mtcnn, resnet, device,
    current_session_id, person_track_mapping, face_person_mapping,
    
    # Основные классы
    VideoRecorder,
    
    # Функции обработки
    initialize_tracker,
    process_tracking,
    process_person_recognition, 
    process_face_recognition_and_linking,
    find_closest_person_in_frame,
    async_save_send_and_record,
    draw_text_with_unicode,
    link_face_to_tracking_id
)

from video_threads import (
    # Qt потоки для GUI
    ROISelector,
    VideoProcessingThread,
    CameraThread
)

import logging
logger = logging.getLogger(__name__)

# Все остальные классы и функции были перенесены в:
# - detection_processing.py (модели, обработка детекции, распознавание)
# - video_threads.py (Qt потоки для GUI)
#
# Этот файл служит точкой входа и обеспечивает обратную совместимость
# с существующим кодом через импорты выше.
#
# Примечание: весь функциональный код был успешно перенесен в новые модули
# для улучшения структуры и поддерживаемости проекта.

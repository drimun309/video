
import sys
from PyQt6.QtWidgets import QApplication
from mainwindow import MainWindow
from network import is_server_active, SERVER_ENABLED
from initialize_db import initialize_database
initialize_database()

if is_server_active():
    print("Сервер активен")
else:
    print("Сервер не доступен")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
 
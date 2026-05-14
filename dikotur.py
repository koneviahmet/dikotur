import sys
import cv2
import os
import time
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QGraphicsBlurEffect,
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QPushButton,
    QFormLayout,
)
from PyQt6.QtCore import QTimer, Qt, QPoint, QSettings
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QFontMetrics
import numpy as np


def app_base_dir():
    """PyInstaller onefile ve normal çalıştırma için uygulama kök dizini."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def camera_backend_candidates():
    """OpenCV VideoCapture için platforma uygun backend sırası."""
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, None]
    if sys.platform == "win32":
        # Windows 10+: çoğu dahili/built-in kamera MSMF ile daha stabil; DSHOW yedek.
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, None]
    return [cv2.CAP_V4L2, None]


def _capture_set_buffer_size(cap, size: int = 1) -> None:
    """DSHOW/MSMF gecikme ve takılma için küçük buffer (destekleniyorsa)."""
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, size)
    except Exception:
        pass


def _capture_warmup(cap, max_attempts: int = 30) -> bool:
    """İlk kareler boş veya siyah olabildiği için birkaç okuma at."""
    for _ in range(max_attempts):
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, "size", 0) > 0:
            return True
    return False


def try_open_camera(index: int):
    """Verilen indeks için uygun backend ile VideoCapture aç; okunabilir kare yoksa None."""
    for backend in camera_backend_candidates():
        try:
            cap = (
                cv2.VideoCapture(index, backend)
                if backend is not None
                else cv2.VideoCapture(index)
            )
        except Exception:
            continue
        if not cap.isOpened():
            continue
        if sys.platform == "win32":
            _capture_set_buffer_size(cap, 1)
        if _capture_warmup(cap):
            return cap
        cap.release()
    return None


def enumerate_camera_indices(max_index: int = 15):
    """Açılabilen kamera indekslerini listeler (her biri kısa süre açılıp kapatılır)."""
    found = []
    for i in range(max_index + 1):
        cap = try_open_camera(i)
        if cap is not None:
            found.append(i)
            cap.release()
    return found


class SettingsDialog(QDialog):
    """Kamera seçimi ayarları."""

    def __init__(self, parent=None, current_index: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.setModal(True)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(280)
        form.addRow("Kamera:", self.combo)
        layout.addLayout(form)

        refresh = QPushButton("Kameraları yenile")
        refresh.clicked.connect(self._populate_cameras)
        layout.addWidget(refresh)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate_cameras()
        idx = self.combo.findData(current_index)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _populate_cameras(self):
        self.combo.clear()
        for cam_index in enumerate_camera_indices():
            self.combo.addItem(f"Kamera {cam_index}", cam_index)
        if self.combo.count() == 0:
            self.combo.addItem("(Kamera bulunamadı)", -1)

    def selected_camera_index(self) -> int:
        data = self.combo.currentData()
        return int(data) if data is not None else -1

# Ses çalma için pygame
try:
    import pygame
    pygame.mixer.init()
    SOUND_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Sound not available: {e}")
    SOUND_AVAILABLE = False

# Try to import posture detector, but don't fail if it's not available
try:
    from posture_detector import PostureDetector
    POSTURE_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Posture detection not available: {e}")
    print("⚠️  Running in camera-only mode")
    POSTURE_AVAILABLE = False

class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Sürükleme için değişkenler
        self.dragging = False
        self.drag_start_position = QPoint()
        
        # Yuvarlak şekil için değişken
        self.is_round = False
        
        # Ekranda yazı gösterme değişkeni (varsayılan: False)
        self.show_posture_text = False
        
        # Duruş uyarı sistemi için değişkenler
        self.poor_posture_start_time = None  # Kötü duruş başlangıç zamanı
        self.sound_played = False  # Ses çalındı mı?
        self.blur_overlay = None  # Blur overlay penceresi
        self.audio_file_path = os.path.join(app_base_dir(), "sesler", "muhammetdikotur.mp3")
        self.settings = QSettings("koneviahmet", "dikotur")
        
        # Duruş tespiti için değişkenler
        if POSTURE_AVAILABLE:
            try:
                self.posture_detector = PostureDetector(show_text=self.show_posture_text)
                self.posture_enabled = True
                print("✅ Posture detection initialized")
            except Exception as e:
                print(f"⚠️  Posture detector initialization failed: {e}")
                self.posture_detector = None
                self.posture_enabled = False
        else:
            self.posture_detector = None
            self.posture_enabled = False
        
        # Pencere boyutları ve konumu
        self.setFixedSize(320, 240)
        
        # Ekran boyutlarını al
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        # Sağ alt köşeye konumlandır
        self.move(screen_width - 320 - 20, screen_height - 240 - 50)
        
        # Pencere özellikleri
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(2, 2, 2, 2)  # Yeşil kenarlık için boşluk
        
        # Kamera görüntüsü için label
        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #00ff00;
                border-radius: 5px;
                background-color: black;
            }
        """)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setWordWrap(True)
        self.camera_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.camera_label.customContextMenuRequested.connect(self._on_camera_context_menu)
        layout.addWidget(self.camera_label)
        
        # Çift tıklama için timer
        self.double_click_timer = QTimer()
        self.double_click_timer.setSingleShot(True)
        self.double_click_timer.timeout.connect(self.toggle_shape)
        self.click_count = 0
        
        self.cap = None
        initial_index = self._saved_camera_index()
        if not self.start_camera(initial_index, save_index=True):
            self._set_camera_placeholder(True)
            QMessageBox.information(
                self,
                "Kamera",
                "Kamera açılamadı.\nÖnizleme alanına sağ tıklayıp Ayarlar'dan kamera seçebilir veya izinleri kontrol edebilirsiniz.",
            )
        
        # Timer ile kamera görüntüsünü güncelle
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

    def _saved_camera_index(self) -> int:
        v = self.settings.value("camera/index", 0)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    def _release_camera(self):
        if getattr(self, "cap", None) is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _set_camera_placeholder(self, active: bool):
        """Kamera yokken etikette kısa yönerge göster."""
        if active:
            self.camera_label.clear()
            self.camera_label.setPixmap(QPixmap())
            self.camera_label.setText("Kamera yok\nSağ tık → Ayarlar")
        else:
            self.camera_label.setText("")

    def start_camera(self, index: int, save_index: bool = True) -> bool:
        """Kamerayı verilen indeksle aç. Başarılıysa True."""
        self._release_camera()
        cap = try_open_camera(index)
        if cap is None:
            self._set_camera_placeholder(True)
            return False
        self.cap = cap
        if save_index:
            self.settings.setValue("camera/index", index)
        if sys.platform == "win32":
            # Birçok Windows sürücüsü 640x480/FPS zorlamasında boş kare veya kilitlenme verir;
            # native çözünürlük + yazılımda resize daha güvenilir.
            _capture_set_buffer_size(self.cap, 1)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("Kamera hazırlanıyor...")
        for i in range(20):
            ret, _ = self.cap.read()
            if ret:
                print(f"✅ Kamera hazır (indeks {index}), {i + 1} kare atlandı")
                break
            print(f"Frame {i + 1} atlandı...")
        self._set_camera_placeholder(False)
        if hasattr(self, "frame_error_count"):
            self.frame_error_count = 0
        return True

    def _on_camera_context_menu(self, pos: QPoint):
        self.open_settings()

    def open_settings(self):
        dlg = SettingsDialog(self, self._saved_camera_index())
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        idx = dlg.selected_camera_index()
        if idx < 0:
            return
        if not self.start_camera(idx, save_index=True):
            QMessageBox.warning(self, "Kamera", "Seçilen kamera açılamadı.")
        
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret and frame is not None:
            try:
                # Görüntüyü küçült
                frame = cv2.resize(frame, (316, 236))  # Kenarlık için biraz küçük
                
                # Yatay simetri (ayna efekti) uygula
                frame = cv2.flip(frame, 1)  # 1 = yatay simetri
                
                # Duruş tespiti yap (eğer etkinse)
                if self.posture_enabled and self.posture_detector:
                    try:
                        # Yazı gösterme durumunu güncelle
                        self.posture_detector.set_show_text(self.show_posture_text)
                        frame, posture_score, is_good_posture = self.posture_detector.analyze_posture(frame)
                        self.update_border_color(is_good_posture)
                        # Duruş uyarı sistemini kontrol et
                        self.check_posture_warnings(is_good_posture)
                    except Exception as e:
                        # Silently disable posture detection on error
                        print(f"⚠️  Posture detection error: {e}")
                        self.posture_enabled = False
                
                # BGR'den RGB'ye çevir
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Windows/MSMF bazen stride bozuk döner; QImage için contiguous gerekir
                rgb_image = np.ascontiguousarray(rgb_image)
                
                # QImage oluştur - PyQt6 için doğru format
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                # PyQt6 için QImage formatını düzelt
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # QPixmap'e çevir ve göster
                pixmap = QPixmap.fromImage(qt_image)
                
                # Pixmap'i label'a ayarla
                if not pixmap.isNull():
                    self.camera_label.setText("")
                    self.camera_label.setPixmap(pixmap)
                    # İlk başarılı görüntüde mesaj yazdır
                    if not hasattr(self, 'first_frame_shown'):
                        print("✅ İlk kamera görüntüsü gösterildi!")
                        self.first_frame_shown = True
                else:
                    print("❌ Pixmap oluşturulamadı!")
            except Exception as e:
                print(f"❌ Frame işleme hatası: {e}")
        else:
            # Sadece ilk birkaç hatada mesaj yazdır
            if not hasattr(self, 'frame_error_count'):
                self.frame_error_count = 0
            self.frame_error_count += 1
            if self.frame_error_count <= 5:
                print(f"❌ Frame okunamadı! ({self.frame_error_count}/5)")
    
    def mousePressEvent(self, event):
        """Mouse basıldığında sürükleme başlat veya çift tıklama kontrolü"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Çift tıklama kontrolü
            self.click_count += 1
            if self.click_count == 1:
                self.double_click_timer.start(300)  # 300ms içinde ikinci tıklama beklenir
            elif self.click_count == 2:
                self.double_click_timer.stop()
                self.toggle_shape()
                self.click_count = 0
                return
            
            # Sürükleme başlat
            self.dragging = True
            self.drag_start_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Mouse hareket ettirildiğinde pencereyi sürükle"""
        if event.buttons() == Qt.MouseButton.LeftButton and self.dragging:
            self.move(event.globalPosition().toPoint() - self.drag_start_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Mouse bırakıldığında sürüklemeyi durdur"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            event.accept()

    def update_border_color(self, is_good_posture):
        """Duruş durumuna göre kenarlık rengini güncelle"""
        if not self.posture_enabled:
            color = "#00ff00"  # Default green when posture detection is disabled
        else:
            color = "#00ff00" if is_good_posture else "#ff0000"  # Yeşil/Kırmızı
        radius = "158px" if self.is_round else "5px"
        
        self.camera_label.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {color};
                border-radius: {radius};
                background-color: black;
            }}
        """)

    def toggle_shape(self):
        """Şekli yuvarlak ve kare arasında değiştir"""
        self.is_round = not self.is_round
        
        # Mevcut duruş durumunu al
        if self.posture_enabled and self.posture_detector:
            posture_status = self.posture_detector.get_posture_status()
            color = "#00ff00" if posture_status['is_good'] else "#ff0000"
        else:
            color = "#00ff00"  # Default green when posture detection is disabled
        
        radius = "158px" if self.is_round else "5px"
        
        self.camera_label.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {color};
                border-radius: {radius};
                background-color: black;
            }}
        """)
        
        shape_text = "🟢 Yuvarlak şekil" if self.is_round else "⬜ Kare şekil"
        print(f"{shape_text} aktif")
        
        self.click_count = 0

    def check_posture_warnings(self, is_good_posture):
        """Duruş uyarı sistemini kontrol eder"""
        current_time = time.time()
        
        if not is_good_posture:
            # Kötü duruş başladı
            if self.poor_posture_start_time is None:
                self.poor_posture_start_time = current_time
                self.sound_played = False
            
            # 3 saniye geçtiyse ve ses çalınmadıysa ses çal
            elapsed = current_time - self.poor_posture_start_time
            if elapsed >= 3.0 and not self.sound_played:
                self.play_warning_sound()
                self.sound_played = True
            
            # 10 saniye geçtiyse blur ekranı göster
            if elapsed >= 10.0:
                if self.blur_overlay is None or not self.blur_overlay.isVisible():
                    self.show_blur_overlay()
        else:
            # İyi duruş - reset
            if self.poor_posture_start_time is not None:
                self.poor_posture_start_time = None
                self.sound_played = False
                # Blur'u kaldır
                if self.blur_overlay is not None and self.blur_overlay.isVisible():
                    self.hide_blur_overlay()
    
    def play_warning_sound(self):
        """Uyarı sesini çalar"""
        if not SOUND_AVAILABLE:
            return
        
        if os.path.exists(self.audio_file_path):
            try:
                pygame.mixer.music.load(self.audio_file_path)
                pygame.mixer.music.play()
                print("🔊 Uyarı sesi çalındı")
            except Exception as e:
                print(f"⚠️  Ses çalma hatası: {e}")
        else:
            print(f"⚠️  Ses dosyası bulunamadı: {self.audio_file_path}")
    
    def show_blur_overlay(self):
        """Tüm ekranı blur yapan overlay penceresini gösterir"""
        if self.blur_overlay is None:
            self.blur_overlay = BlurOverlayWindow()
        self.blur_overlay.show()
        self.blur_overlay.raise_()  # En üste getir
        self.blur_overlay.activateWindow()  # Aktif hale getir
        print("🌫️  Ekran blur aktif")
    
    def hide_blur_overlay(self):
        """Blur overlay penceresini gizler"""
        if self.blur_overlay is not None:
            self.blur_overlay.hide()
            print("✅ Ekran blur kaldırıldı")

    def closeEvent(self, event):
        # Kamera kaynağını serbest bırak
        self._release_camera()
        
        # Duruş tespiti kaynaklarını temizle
        if hasattr(self, 'posture_detector') and self.posture_detector is not None:
            try:
                self.posture_detector.cleanup()
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")
        
        # Blur overlay'i kapat
        if hasattr(self, 'blur_overlay') and self.blur_overlay is not None:
            self.blur_overlay.close()
        
        # Ses sistemini temizle
        if SOUND_AVAILABLE:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except:
                pass
        
        event.accept()


class BlurOverlayWindow(QMainWindow):
    """Tüm ekranı blur yapan overlay penceresi"""
    def __init__(self):
        super().__init__()
        
        # Tam ekran yap
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
        # Pencere özellikleri - en üstte, tıklanamaz, şeffaf arka plan
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Merkez widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Güçlü blur efekti ekle
        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(50)  # Blur miktarını artırdık (15'ten 50'ye)
        self.central_widget.setGraphicsEffect(blur_effect)
        
        # Yarı şeffaf beyaz overlay (blur ile birlikte flu görünüm)
        self.central_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.6);
            }
        """)
    
    def paintEvent(self, event):
        """Ek overlay efekti ve yazı için paint event"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Ek yarı şeffaf katman (daha flu görünüm için)
        painter.fillRect(self.rect(), QColor(255, 255, 255, 80))
        
        # Ortaya "Dik Oturmalısın" yazısını ekle
        text = "Dik Oturmalısın"
        font = QFont("Arial", 48, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 0, 0))  # Kırmızı renk
        
        # Metni ortala
        font_metrics = QFontMetrics(font)
        text_rect = font_metrics.boundingRect(text)
        text_x = (self.width() - text_rect.width()) // 2
        text_y = (self.height() + text_rect.height()) // 2
        
        # Gölge efekti için siyah outline
        painter.setPen(QColor(0, 0, 0))
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]:
            painter.drawText(text_x + dx, text_y + dy, text)
        
        # Ana yazı (kırmızı)
        painter.setPen(QColor(255, 0, 0))
        painter.drawText(text_x, text_y, text)

def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("koneviahmet")
    app.setApplicationName("dikotur")

    # Ana pencere
    window = CameraWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
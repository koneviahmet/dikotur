import sys
import cv2
import os
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QMessageBox, QGraphicsBlurEffect
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QPainter, QPainterPath, QBrush, QColor, QFont, QFontMetrics
import numpy as np

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
        self.audio_file_path = os.path.join(os.path.dirname(__file__), "sesler", "muhammetdikotur.mp3")
        
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
        layout.addWidget(self.camera_label)
        
        # Çift tıklama için timer
        self.double_click_timer = QTimer()
        self.double_click_timer.setSingleShot(True)
        self.double_click_timer.timeout.connect(self.toggle_shape)
        self.click_count = 0
        
        # Kamera başlat - macOS için farklı backend'ler dene
        self.cap = None
        
        # Önce AVFoundation dene
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret:
                    print("✅ AVFoundation backend çalışıyor")
                else:
                    self.cap.release()
                    self.cap = None
        except:
            self.cap = None
        
        # AVFoundation çalışmazsa varsayılan backend'i dene
        if self.cap is None:
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret:
                        print("✅ Varsayılan backend çalışıyor")
                    else:
                        self.cap.release()
                        self.cap = None
            except:
                self.cap = None
        
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.critical(self, "Hata", "Kamera açılamadı! Lütfen kamera izinlerini kontrol edin.")
            return
        
        # Kamera ayarları
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Kamera hazır olduğundan emin ol - daha fazla frame atla
        print("Kamera hazırlanıyor...")
        for i in range(20):  # Daha fazla frame atla
            ret, _ = self.cap.read()
            if ret:
                print(f"✅ Kamera hazır! {i+1} frame atlandı")
                break
            else:
                print(f"Frame {i+1} atlandı...")
        
        # Timer ile kamera görüntüsünü güncelle
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
        
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
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # QImage oluştur - PyQt6 için doğru format
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                # PyQt6 için QImage formatını düzelt
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # QPixmap'e çevir ve göster
                pixmap = QPixmap.fromImage(qt_image)
                
                # Pixmap'i label'a ayarla
                if not pixmap.isNull():
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
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
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
    
    # Ana pencere
    window = CameraWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
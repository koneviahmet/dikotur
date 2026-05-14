#!/bin/bash

# macOS için uygulamayı çalıştırma scripti
# Gerekli paketleri kontrol et ve yükle

echo "Gerekli paketler kontrol ediliyor..."

# Python paketlerini kontrol et
python3 -c "import PyQt6, cv2, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Gerekli paketler yükleniyor..."
    pip3 install -r requirements.txt
fi

echo "Uygulama başlatılıyor..."
python3 dikotur.py

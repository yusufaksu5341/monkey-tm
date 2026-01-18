import cv2

# Kamera açma
cap = cv2.VideoCapture(0)

# Kamera ayarlarını kontrol et
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Video çerçevesi genişlik ve yüksekliği
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Kamera çözünürlüğü: {frame_width}x{frame_height}")

while True:
    # Kameradan bir frame oku
    ret, frame = cap.read()
    
    if not ret:
        print("Frame okunamadı!")
        break
    
    # Frame'i göster
    cv2.imshow('Kamera Görüntüsü', frame)
    
    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Çıkılıyor...")
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

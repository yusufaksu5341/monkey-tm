import cv2
import numpy as np
import os
np.set_printoptions(suppress=True)
import tf_keras as keras
print("Model yükleniyor...")
model = keras.models.load_model("keras_model.h5", compile=False)
print("Model yüklendi!")
class_names = open("labels.txt", "r", encoding='utf-8').readlines()
images_dict = {}
image_extensions = ['.jpg', '.jpeg', '.png']
images_folder = 'images'
for line in class_names:
    class_name = line.strip().split(' ', 1)[1]
    for ext in image_extensions:
        image_path = os.path.join(images_folder, f"{class_name}{ext}")
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                images_dict[class_name] = cv2.resize(img, (300, 300))
                print(f"Yüklendi: {image_path}")
                break
camera = cv2.VideoCapture(0)
last_predicted_class = None
prediction_counter = 0
min_predictions = 2
confidence_threshold_default = 0.7
class_sensitivity = {'fikriolan': {'min_predictions': 1, 'confidence': 0.50}}
default_image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(default_image, "Bekliyor...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
current_display_image = default_image.copy()
while True:
    ret, image = camera.read()
    if not ret:
        break
    image = cv2.flip(image, 1)
    camera_frame = cv2.resize(image, (640, 480))
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_normalized = (image_array / 127.5) - 1
    prediction = model.predict(image_normalized, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    predicted_class = class_name.strip().split(' ', 1)[1]
    if predicted_class == last_predicted_class:
        prediction_counter += 1
    else:
        last_predicted_class = predicted_class
        prediction_counter = 1
    min_req = class_sensitivity.get(predicted_class, {}).get('min_predictions', min_predictions)
    conf_req = class_sensitivity.get(predicted_class, {}).get('confidence', confidence_threshold_default)
    if prediction_counter >= min_req and confidence_score > conf_req:
        if predicted_class in images_dict:
            current_display_image = cv2.resize(images_dict[predicted_class], (640, 480))
    if predicted_class == 'fikriolan':
        status_text = f"fikriolan boost: cnt={prediction_counter}/{min_req}, conf={confidence_score:.2f}/{conf_req:.2f}"
        cv2.putText(camera_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    print(f"Class: {predicted_class}, Confidence: {np.round(confidence_score * 100)}%, Counter: {prediction_counter}")
    text = f"{predicted_class} ({confidence_score:.2f})"
    cv2.putText(camera_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    combined_frame = np.hstack((camera_frame, current_display_image))
    cv2.imshow("Kamera | Tespit Edilen", combined_frame)
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27 or keyboard_input == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

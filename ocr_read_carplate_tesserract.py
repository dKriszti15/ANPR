import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = "D:\\CarplateRecognition\\sandbox_cameraTest\\cropped_boxes\\carplate_0_1_conf_0.85.jpg"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

scale_percent = 300
width = int(gray.shape[1] * scale_percent / 100)
height = int(gray.shape[0] * scale_percent / 100)
resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

blur = cv2.bilateralFilter(resized, 11, 17, 17)

_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# A-Z + 0-9
custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

text = pytesseract.image_to_string(thresh, config=custom_config)

print("Felismert szoveg:", text.strip()[:8])

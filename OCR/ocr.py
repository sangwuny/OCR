from PIL import Image
import pytesseract
import cv2
import numpy as np
import os

# 1. Tesseract 경로 설정
# Windows에서는 아래 경로를 Tesseract 설치 경로로 설정해야 합니다.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 2. 이미지 로드 및 전처리
def preprocess_image(image_path):
    # OpenCV로 이미지 읽기
    image = cv2.imread(image_path)
    # Grayscale 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 이진화 (Binary Thresholding)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return binary

# 3. OCR 수행
def perform_ocr(image_path):
    processed_image = preprocess_image(image_path)
    # OCR 수행
    text = pytesseract.image_to_string(processed_image, lang="eng")  # 'lang'는 사용할 언어
    return text

# 4. 테스트 실행
if __name__ == "__main__":
    # 현재 스크립트 위치 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 상대경로로 이미지 경로 설정
    image_path = os.path.join(script_dir, "image", "img3.png")
    extracted_text = perform_ocr(image_path)
    print("Extracted Text:")
    print(extracted_text)

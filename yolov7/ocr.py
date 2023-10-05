import cv2
import numpy as np

OCR_TYPE = "EASY_OCR"

if OCR_TYPE == "PADDLE_OCR":
    from paddleocr import PaddleOCR
    PADDLE_OCR = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='en', det=False, show_log=False)
    reader = PADDLE_OCR
elif OCR_TYPE == "EASY_OCR":
    import easyocr
    EASY_OCR = easyocr.Reader(['en'], gpu=True)
    OCR_TH = 0.2
    reader = EASY_OCR

def read_LP(plate):
    print(OCR_TYPE)
    if OCR_TYPE == "PADDLE_OCR":
        result = reader.ocr(plate, cls=False, det=True)

        text = ''

        if result[0] is not None:
            for i in range(len(result[0])):
                text += result[0][i][1][0]

        return text
    elif OCR_TYPE == "EASY_OCR":
        ocr_result = reader.readtext(plate)

        text = filter_text(plate, ocr_result, OCR_TH)

        if len(text) == 1:
            text = text[0].upper()

        return text

def filter_text(region, ocr_result, region_threshold):
    rectangle_area = region.shape[0] * region.shape[1]

    plate = []

    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_area > region_threshold:
            plate.append(result[1])

    return ''.join(plate)


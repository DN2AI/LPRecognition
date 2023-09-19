import easyocr
import cv2
import numpy as np

EASY_OCR = easyocr.Reader(['en'], gpu=True)
OCR_TH = 0.2

def recognize_plate_easyocr(plate, reader=EASY_OCR):
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
    

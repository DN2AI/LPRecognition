import cv2
from ocr import read_LP

try:
    from sort.sort import Sort

    USE_SORT_TRACKER = True
    sort_tracker = Sort()
except:
    USE_SORT_TRACKER = False

license_plate = {}

def get_image_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def read_license_plate(source_img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return ''

    # Crop image
    LP_cropped = source_img[int(y1) : int(y2), int(x1) : int(x2), :]

    license_plate_text = read_LP(LP_cropped)

    return license_plate_text


def check_format_number_license_plate(license_plate_text):
    ordinal_number_limit = 5 if '.' in license_plate_text else 4

    license_plate_text = license_plate_text.replace('-', '').replace('.', '')
    local_number = license_plate_text[:2]
    seri_number = license_plate_text[2:4]
    ordinal_number = license_plate_text[4:]

    if len(local_number) < 2 or len(seri_number) < 2 or len(license_plate_text) > 9:
        return False

    if local_number.isdigit() and (
        (seri_number[0].isalpha() and seri_number[1].isdigit()) or seri_number.isalpha()
    ):
        if len(ordinal_number) == ordinal_number_limit and ordinal_number.isdigit():
            return True
    return False

def assign_number_license_plate(LP_id, license_plate_text, LP_cropped):
    if LP_id not in license_plate:
        license_plate[LP_id] = {'text': '', 'sharpness': 0}

    if check_format_number_license_plate(license_plate_text):
        LP_sharpness = get_image_sharpness(LP_cropped)
        if LP_sharpness >= license_plate[LP_id]['sharpness']:
            license_plate[LP_id]['text'] = license_plate_text
            license_plate[LP_id]['sharpness'] = LP_sharpness


import cv2
import numpy as np

from Preprocess import preprocess, Hough_transform, rotation_angle, rotate_LP

def crop_n_rotate_LP(source_img, x1, y1, x2, y2):
    w = int(x2 - x1)
    h = int(y2 - y1)
    ratio = w / h
    # print ('ratio',ratio)
    if 0.8 <= ratio <= 1.5 or 3.5 <= ratio <= 6.5:
        cropped_LP = source_img[int(y1):int(y2), int(x1):int(x2), :]
        cropped_LP_copy = cropped_LP.copy()

        imgGrayscaleplate, imgThreshplate = preprocess(cropped_LP)
        canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=2)

        linesP = Hough_transform(dilated_image, nol=6)
        for i in range(0, len(linesP)):
            l = linesP[i][0].astype(int)

        angle = rotation_angle(linesP)
        rotate_thresh = rotate_LP(imgThreshplate, angle)
        LP_rotated = rotate_LP(cropped_LP, angle)
    else:
        angle, rotate_thresh, LP_rotated = None, None, None

    return angle, rotate_thresh, LP_rotated


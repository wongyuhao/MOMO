import cv2

"""
Apply adapting thresholding on an image
Args:
    image: cv2 formatted image binary 

Returns:
    thresh: input image with adaptive thresholding applied
"""


def adaptive_thresholing(image):
    # https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    return thresh

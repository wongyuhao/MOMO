import re

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output

from skimage.filters import threshold_local
from PIL import Image

# take an image



# input: image file_name
# downscales image and converts it to grayscale


# runs canny edge detection on a preprocessed image
def find_receipt(image, resize_ratio):
    def preprocess_image(image, ratio):
        width = int(image.shape[1] * ratio)
        height = int(image.shape[0] * ratio)

        dimensions = (width, height)

        resized = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
        grayscaled = cv2.cvtColor(resized.copy(), cv2.COLOR_BGR2GRAY)  # convert image to grayscale
        # inter_area interpolation is better for shrinking images
        return resized, grayscaled

    def find_receipt_contour(contours):
        # Use contour approximation to find the bounding box of the receipt
        # https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
            if len(approx) == 4:
                return approx

    resized, grayscaled = preprocess_image(image, resize_ratio)

    # blur image to reduce noise
    smoothed = cv2.GaussianBlur(grayscaled, ksize=(3, 3), sigmaX=0)

    # detect potential regions of interest in the image,
    # In this case, white regions that may be the receipt
    # we can use dilation:
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel = np.ones((9, 9))
    dilation = cv2.dilate(smoothed, kernel, iterations=2)

    # run canny edge detection on the image:
    edges = cv2.Canny(dilation, 120, 180, apertureSize=3)

    # erode and dilate to connect broken edges
    d2 = cv2.dilate(edges, kernel, iterations=3)
    d3 = cv2.erode(d2, kernel, iterations=3)

    # cv2.imshow('edges (eroded + dilated)', d3)

    # find contours
    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(d3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the receipt contour from the 10 largest contours in the image
    res_contour = find_receipt_contour(sorted(contours, key=cv2.contourArea, reverse=True)[:10])

    res_image = cv2.drawContours(resized.copy(), [res_contour], -1, (0, 0, 255), 1)

    return res_image, res_contour


def perspective_warp(img, contour, ratio):
    def euclidian_distance(pointA, pointB):
        return np.sqrt(
            (np.power(pointA[0] - pointB[0], 2)) +
            (np.power(pointA[1] - pointB[1], 2))
        )

    def contour_to_rect(contour, ratio):
        pts = contour.reshape(4, 2)  # reshape to flatten nested points

        rect = np.zeros((4, 2), dtype="float32")
        # top left = min_x and min_y => min summed pair
        rect[0] = min(pts, key=lambda x: abs(x[0] + x[1]))
        # top right = max_x and min_y => min difference pair
        rect[1] = min(pts, key=lambda x: x[1] - x[0])
        # bottom right = max_x and max_y => max summed pair
        rect[2] = max(pts, key=lambda x: abs(x[0] + x[1]))
        # bottom left = max_x and max_y => max difference pair
        rect[3] = max(pts, key=lambda x: x[1] - x[0])

        # upscale to reverse original scaling ratio
        return rect / ratio

    original_rect = contour_to_rect(contour, ratio)
    (top_left, top_right, bottom_right, bottom_left) = original_rect

    w_top = euclidian_distance(bottom_left, bottom_right)
    w_bottom = euclidian_distance(top_left, top_right)

    h_left = euclidian_distance(top_left, bottom_left)
    h_right = euclidian_distance(top_right, bottom_right)

    maxWidth = max(int(w_top), int(w_bottom))
    maxHeight = max(int(h_left), int(h_right))

    # unwarpped corners
    mapped = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # transform from orig to mapped
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(original_rect, mapped), (maxWidth, maxHeight))

def adaptive_thresholing(image):
    # https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    return thresh

def format_dollar_value(value):
    # Match the dollar value using regex
    match = re.match(r'^(\$?\d+)\.(\d+)$|^(\$?\d+)$|^\.(\d+)$|^(\d+)\.(\d+)$|^(\d+)$', value)
    if match:
        # Extract the numbers before and after the decimal point
        groups = match.groups()
        dollars = int(groups[0] or groups[2] or groups[4] or groups[5] or groups[6] or 0)
        cents = int(groups[1] or groups[3] or 0)
        # Format the numbers as xx.xx
        return "{:02d}.{:02d}".format(dollars, cents)
    else:
        # Return the original value if it doesn't match the expected format
        return value

def parse_item_lines(lines):
    regex = r'^(\w+(?:[\s+\w+~.])*)\s+((\$?\d*(?: *\.\d+)?|\$?\d*(?:\.\d+ *)?))$'

    res = []
    for line in lines:
        parsed_line = re.match(regex, line)
        if parsed_line:
            name = parsed_line.group(1)
            value = parsed_line.group(2)

            value = format_dollar_value(value.removeprefix('$'))

            item = (name, value)
            print(item)
            print('\n')


            res.append(item)
    return res

def detect_text(image, lib=None):
    if lib == 'TESSERACT':
        return detect_text_with_tesseract(image)
    else:
        print('OCR library not specified. Running Tesseract.')
        return detect_text_with_tesseract(image)

def detect_text_with_tesseract(image):
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['level'])
    boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    lines = pytesseract.image_to_string(image)

    result = list(filter(lambda x: len(x) != 0, lines.splitlines()))

    res = parse_item_lines(result)
    return boxes, res


def pipeline(image):

    # step 0: determine resize ratio based on image dimensions
    resize_ratio = 500 / image.shape[0]

    # step 1: find and highlight receipt in the image
    receipt_image, receipt_contour = find_receipt(image, resize_ratio)
    cv2.imshow('boxed receipt', receipt_image)

    # step 2: warp the highlighted region into a flat receipt
    warped_receipt = perspective_warp(image.copy(), receipt_contour, resize_ratio)

    # step 3: apply adaptive thresholding to the image as preprocessing for tesseract
    bw_receipt = adaptive_thresholing(warped_receipt)
    cv2.imshow("Mean Adaptive Thresholding", bw_receipt)

    # step 4: identify text boxes with tesseract
    boxes, result = detect_text(bw_receipt)
    cv2.imshow('box', boxes)

    print(result)
    cv2.waitKey(0)
    return


file1 = "images/7.jpg"
file2 = "images/6.jpg"
file3 = 'images/tj2.png'



img = cv2.imread(file3)
pipeline(img)

import cv2
import numpy as np

"""
Finds the receipt area within a given image

Args:
    image: cv2 formatted image binary
    resize_ratio: image downscale ratio
    
Returns:
    (res_image, res_contour, edges)
    res_image: cv2 formatted image with receipt contour highlighted
    res_contour: contour information of the receipt area
    edges: cv2 formatted image of Canny image detection on the input
"""


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
    raw_edges = cv2.Canny(dilation, 120, 180, apertureSize=3)

    # erode and dilate to connect broken edges
    dilated_edges = cv2.dilate(raw_edges, kernel, iterations=3)
    edges = cv2.erode(dilated_edges, kernel, iterations=3)

    # find contours
    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the receipt contour from the 10 largest contours in the image
    res_contour = find_receipt_contour(sorted(contours, key=cv2.contourArea, reverse=True)[:10])

    res_image = cv2.drawContours(resized.copy(), [res_contour], -1, (0, 0, 255), 1)

    return res_image, res_contour, edges

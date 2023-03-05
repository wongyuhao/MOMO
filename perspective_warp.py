import cv2
import numpy as np

"""
Given a contour over the area of interest, apply perspective warp to flatten this area from the original image

Args:
    image: cv2 formatted image binary 
    resize_ratio: image downscale ratio

Returns:
    warped_image: cv2 formatted flattened image of the receipt
"""


def perspective_warp(image, contour, resize_ratio):
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

    original_rect = contour_to_rect(contour, resize_ratio)
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
    warped_image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(original_rect, mapped), (maxWidth, maxHeight))
    return warped_image

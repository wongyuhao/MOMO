import re
import time
import uuid

import numpy as np
import cv2
from PIL import Image

from find_receipt import find_receipt
from perspective_warp import perspective_warp
from text_detection import detect_text
from threshold import adaptive_thresholing
from s3_util import uploadPILtoBucket


def pipeline(image):
    DEBUG = False

    # step 0: determine resize ratio based on image dimensions
    resize_ratio = 500 / image.shape[0]

    start = time.process_time()

    # step 1: find and highlight receipt in the image
    receipt_image, receipt_contour, edges_image = find_receipt(image, resize_ratio)
    if DEBUG:
        cv2.imshow('boxed receipt', receipt_image)
        cv2.imshow('edges', edges_image)

    # step 2: warp the highlighted region into a flat receipt
    warped_image = perspective_warp(image.copy(), receipt_contour, resize_ratio)
    if DEBUG:
        cv2.imshow('warped_receipt', warped_image)

    # step 3: apply adaptive thresholding to the image as preprocessing for tesseract
    bw_receipt_image = adaptive_thresholing(warped_image)
    if DEBUG:
        cv2.imshow("Mean Adaptive Thresholding", bw_receipt_image)

    # step 4: identify text boxes with tesseract
    boxed_text_image, items = detect_text(bw_receipt_image)

    pipeline_elapsed = time.process_time() - start

    if DEBUG:
        cv2.imshow('box', boxed_text_image)
        cv2.waitKey()

    pipeline_steps = {
        'edges': edges_image,
        'contour': receipt_image,
        'perspective': warped_image,
        'threshold': bw_receipt_image,
        'scanned': boxed_text_image
    }

    urls = []

    # only save to s3 if we find text, otherwise return empty
    if len(items) > 0:
        id = str(uuid.uuid4())
        for step, img in pipeline_steps.items():
            # convert openCV image to PIL image
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((500, 500), Image.LANCZOS)
            urls.append(uploadPILtoBucket(pil_img, 'temp/%s/%s.jpeg' % (id, step)))

    total_elapsed = time.process_time() - start

    res = {
        "items": items,
        "images": urls,
        "pipeline_elapsed": "%.4f" % pipeline_elapsed,
        "total_elapsed": "%.4f" % total_elapsed
    }
    return res

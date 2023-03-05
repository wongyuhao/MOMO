import re

import cv2
import pytesseract
from pytesseract import Output


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

            item = {"name": name, "value": value}
            res.append(item)
    return res
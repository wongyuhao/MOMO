# MOMO - Bill Splitting Made Easy

MOre MOney (MOMO) is a smart bill splitting utility that removes the awkwardness of managing shared bills in social settings. 
MOMO won [2nd Place at DubHacks 2022](https://devpost.com/software/momo-sjxg21), the largest collegiate hackathon in the Pacific Northwest! 

## About
This repository contains the codebase for the backend service for MOMO that handles image ingestion, processing and parsing using OpenCV and Tesseract OCR,
running on a Flask server. Additional assets are stored in AWS S3 buckets, including raw and processed images. Computer vision techniques used here include Warping and Convolutions, Morphological Dilation and Erosion, Canny Edge Detection, Contour Approximation and more.

In production, this server is containerized and hosted on AWS Lightsail, with the Web interface and proof of concept hosted at https://www.haoyudoing.com/cv.




![Screenshot 2023-06-14 at 12-59-12 https __www haoyudoing com](https://github.com/wongyuhao/MOMO/assets/54234367/077ed803-e31f-40d7-b438-98a50e2e5ac8)

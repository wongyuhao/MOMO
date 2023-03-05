import io
import os

from dotenv import load_dotenv

import boto3, botocore
from PIL import Image

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)


def uploadPILtoBucket(image, filename):
    BUCKET = os.getenv('AWS_BUCKET_NAME')
    ENDPOINT = os.getenv('AWS_DOMAIN')

    temp_file = io.BytesIO()
    image.save(temp_file, format="JPEG")
    temp_file.seek(0)

    s3.upload_fileobj(
        Fileobj=temp_file,
        Bucket=BUCKET,
        Key=filename,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    return '%s/%s' % (ENDPOINT, filename)

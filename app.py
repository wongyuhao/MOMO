import io

import cv2
from PIL import Image
from flask import request, send_file, jsonify, Response
import flask
import numpy as np
import json

from flask_cors import CORS, cross_origin

from main import pipeline

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.post('/cv')
@cross_origin()
def intake_image():
    file = request.files['file']
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    try:
        res = pipeline(frame)
        return jsonify(res)
    except:
        return Response(
            'The model failed on your image',
            400
        )



if __name__ == '__main__':
    app.run()

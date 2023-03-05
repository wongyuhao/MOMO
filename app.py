import cv2
from flask import request, jsonify, Response
import flask
import numpy as np

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
    try:
        file = request.files['file']
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        res = pipeline(frame)

        return jsonify(res)
    except Exception as e:
        return Response(
            str(e),
            400
        )


if __name__ == '__main__':
    app.run()

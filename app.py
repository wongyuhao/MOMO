
import cv2
from flask import request, jsonify, Response
import flask
import numpy as np


from main import pipeline

app = flask.Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.post('/cv')
def intake_image():
    file = request.files['file']
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    try:
        res = pipeline(frame)
        return jsonify(res)
    except Exception as e:
        return Response(
            str(e),
            400
        )



if __name__ == '__main__':
    app.run()

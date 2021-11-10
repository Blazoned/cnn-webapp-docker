import os

from flask_socketio import SocketIO, send, emit
from flask import Flask, render_template, request
import logging

from imageprocessor import ImageProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

socket_io = SocketIO(app, always_connect=True, logger=False, engineio_logger=False)

logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('socketio').setLevel(logging.INFO)
logging.getLogger('engineio').setLevel(logging.INFO)


@app.route('/')
def index():
    """Home page, normal camera test."""
    return render_template('index.html')


@socket_io.on('connect')
def connect_webcam():
    app.logger.info(f'Web client connected: {request.sid}')


@socket_io.on('snapshot-upload')
def image_upload(blob):
    app.logger.info(f'Processing webcam feed for user: {request.sid}')

    img_processor = ImageProcessor(blob)
    img = img_processor.get_image_source(force_processing=True)
    emit('update-data', img_processor.predictions)

    return img


@socket_io.on('disconnect')
def disconnect_webcam():
    app.logger.info(f'Web client disconnected: {request.sid}')


def main():
    socket_io.run(app=app, host='0.0.0.0', port=5252)


def get_app():
    return socket_io.run(app=app, host='0.0.0.0', port=5252)


if __name__ == '__main__':
    main()

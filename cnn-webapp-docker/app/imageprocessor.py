import cv2
import numpy as np

from models import FacialExpressionModel


class ImageProcessor:
    face_cascade = cv2.CascadeClassifier('cascades/frontalface_default_haarcascade.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    model = FacialExpressionModel('model_data/model.json', 'model_data/model_weights.oversampled.hdf5')

    def __init__(self, binary_blob: str):
        self._blob = binary_blob
        self._decoded = None
        self.predictions = []

    def process(self):
        self._decoded = cv2.imdecode(np.frombuffer(self._blob, np.uint8), -1)

        grey_frame = cv2.cvtColor(self._decoded, cv2.COLOR_RGBA2GRAY)
        faces = self.face_cascade.detectMultiScale(grey_frame, 1.3, 5)

        for (x, y, width, height) in faces:
            face = cv2.resize(grey_frame[y:y+height, x:x+width], (48, 48))
            preds = self.model.predict(face[np.newaxis, :, :, np.newaxis])
            self.predictions.append(preds)

            # text_first_place = f'{preds[0][0]}: {(preds[0][1]*100):3.1f}%'

            cv2.putText(self._decoded, preds[0][0], (x, y-10), self.font, .85, (47, 47, 255), 2)
            cv2.rectangle(self._decoded, (x, y), (x+width, y+height), (192, 192, 0), 1)

    def get_image_source(self, force_processing: bool = False) -> str:
        if self._decoded:
            return cv2.imencode('.jpg', self._decoded)[1].tostring()
        else:
            if force_processing:
                self.process()
                return cv2.imencode('.jpg', self._decoded)[1].tostring()
            else:
                return self._blob

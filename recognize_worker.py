"""
recognize_worker.py
────────────────────
Long-running Python worker process.
Keeps MediaPipe loaded in memory for fast per-frame inference.

Protocol (one JSON line per frame):
  STDIN  ← {"image": "data:image/jpeg;base64,/9j/..."}
  STDOUT → {"gesture":"V / 2","confidence":0.85,"hands":[[...],[...]],"landmarks":[...],"confirmed":true,"num_hands":2}
"""

import sys
import os
import json
import base64

import numpy as np
import cv2

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from gesture_model import GestureRecognizer
from preprocessing import Preprocessor

recognizer   = GestureRecognizer()
preprocessor = Preprocessor(target_size=(640, 480))

print("[worker] Ready.", file=sys.stderr, flush=True)


def decode_image(data_url: str) -> np.ndarray:
    _, b64 = data_url.split(',', 1)
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req   = json.loads(raw_line)
            frame = decode_image(req['image'])
            frame = preprocessor.process(frame)
            result = recognizer.predict(frame)
            result.setdefault('hands', [])
            sys.stdout.write(json.dumps(result) + '\n')
            sys.stdout.flush()

        except Exception as e:
            error_resp = {
                'gesture':    '',
                'confidence': 0.0,
                'hands':      [],
                'landmarks':  [],
                'hints':      [],
                'hand_states':[],
                'confirmed':  False,
                'num_hands':  0,
                'error':      str(e)
            }
            sys.stdout.write(json.dumps(error_resp) + '\n')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
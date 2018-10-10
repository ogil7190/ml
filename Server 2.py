from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
import base64
import Recognizer
from threading import Thread
import cv2

OPEN = True
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ogil-jax'
socket = SocketIO(app)

@socket.on('connect')
def handle_connect():
    print('connected')
    
@socket.on('on-data')
def handle_data(data):
    img = readbase(data)
    ans = recognizer.predict(img)
    print(ans)
    emit('completed', ans)

def showUI():
    global OPEN
    DROP_RATE = 10
    count = 0
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        cv2.imshow('Video', frame)
        if ret: 
            if(count > DROP_RATE):
                count = 0
                if(OPEN):
                    OPEN = False
                    thread = Thread(target=recognize, args=(frame, ))
                    thread.start()
            else:
                count += 1
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

def recognize(frame):
    recognizer.facedetect(frame, reset_open)

def reset_open():
    global OPEN
    OPEN = True

def onReady():
    print('Something')
    #socketio.run(app, host='', port=7190)

def readbase(base_string):
    dope = base64.b64decode(base_string)
    filename = 'img.jpg'
    with open(filename, 'wb') as f:
        f.write(dope)
    return filename

if __name__ == '__main__':
    recognizer = Recognizer.Recognizer()
    showUI()

from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
import base64
import Recognizer
from threading import Thread

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ogil-jax'
socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print('connected')
    
@socketio.on('on-data')
def handle_data(data):
    img = readbase(data)
    ans = recognizer.predict(img)
    print(ans)
    emit('completed', ans)

def onReady():
    print('Everything is ready')

def readbase(base_string):
    dope = base64.b64decode(base_string)
    filename = 'img.jpg'
    with open(filename, 'wb') as f:
        f.write(dope)
    return filename

if __name__ == '__main__':
    recognizer = Recognizer.Recognizer(onReady)
    socketio.run(app, host='', port=7190)
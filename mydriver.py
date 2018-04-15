# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 10:54:12 2018

@author: dev
"""

import numpy as np

import argparse
import socketio
import eventlet
import base64

from keras.models import load_model
from flask import Flask
from PIL import Image
from io import BytesIO
from helper import preprocess

socket = socketio.Server()

app = Flask(__name__)

model = None

max_speed = 16
min_speed = 6
speed_limit = max_speed

  
@socket.on('connect')
def connect(sid, environ):
    print('connected: ', sid)
    send_control(0, 0)
    
def send_control(angle, throttle):
    socket.emit("steer",
                data={
                        'steering_angle' : angle.__str__(),
                        'throttle' : throttle.__str__()
                },
                skip_sid=True)

@socket.on('telemetry')
def telemetry(sid, data):
    
    if data:
        
        angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        
        try:
            image = np.asarray(image)
            image = preprocess(image)
            image = np.array([image])
            
            angle = float(model.predict(image, batch_size=1))
			
            global speed_limit
            if speed > speed_limit:
                speed_limit = min_speed
            else:
                speed_limit = max_speed
                
            throttle = 1.0 - ((angle**2)*1) - (speed/speed_limit)**2
            throttle = throttle * 0.5
            print('angle: {} throttle: {}  speed: {}'.format(angle, throttle, speed))
            send_control(angle, throttle)
            
        except Exception as e:
            print(e)
            
    else:
        socket.emit('manual', data={}, skip_sid=True)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='path to model h5 file')
    #parser.add_argument('images', type=str, nargs='?', default='', help='path to images')
    
    args = parser.parse_args()
    model = load_model(args.model)
    
    app = socketio.Middleware(socket, app)
    
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
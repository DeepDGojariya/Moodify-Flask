from flask import Flask,Response,redirect,url_for
from flask.templating import render_template
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import sqlite3
from sqlite3 import Error

app = Flask(__name__)
saved_model = load_model('model_2.h5')
print("model loaded")

opDict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
lst=[]

def gen_frames():  
    camera = cv2.VideoCapture(0)
    global lst
    i=0
    while True and i<90:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            gray = frame #future use line 25
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            #start prediction
            gray = cv2.cvtColor(gray,code=cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,(48,48))
            result = saved_model.predict(gray[np.newaxis,:,:,np.newaxis])
            result = list(result[0])
            img_index = result.index(max(result))
            lst.append(opDict[img_index])
            #end prediction
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            i+=1
    camera.release()
    
            
         






@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/playlist')
def create_playlist():
    global lst
    playlist = []
    print(lst)
    max_op = max(set(lst[-20:]),key=lst.count)#max output class
    #connection with database
    conn = None
    try:
        conn = sqlite3.connect('playlist.db')
    except Error as e:
        print(e)
    
    cur = conn.cursor()
    cur.execute("SELECT * FROM %s" %(max_op))
    rows = cur.fetchall()
    for row in rows:
        playlist.append({'song':row[0],
                'artist':row[1],
                'url':row[2]})

    cur.close()#closing the cursor
    del cur#deleting the cursor
    
    context={'emotion':max_op,'playlist':playlist}
    return render_template('playlist.html',context=context)

@app.route('/about')
def about():
    return render_template('about.html')






if __name__ == '__main__':
    app.run(debug=True)
    
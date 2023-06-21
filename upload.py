from flask import Flask, request,render_template
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
trained_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
@app.route('/index', methods=['GET'])
def index():
    return render_template('uploadVideo.html')

@app.route('/upload', methods=['POST'])
async def upload():
  
        if 'videoFile' not in request.files:
            return 'No video file provided'

        video_file = request.files['videoFile']
        if video_file.filename == '':
            return 'No selected video file'

        # Save the video file to a desired location
      
        video_path = 'videos/' + video_file.filename
        video_file.save(video_path)

        # 
        video = cv2.VideoCapture(video_path)
        v_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("video len",  v_len)

        faceCordinates = []
        for i in range(100):
            successfully_frame_read, frame = video.read()
            grayScaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faceCordinates.append(trained_face.detectMultiScale(grayScaled))


        print("the length of the faceCordinates" , len(faceCordinates))

        faceL = []
        for i in range(len(faceCordinates)):
            for (x,y,w,h) in faceCordinates[i]:
            
                theFace = frame[y:y+h, x:x+w]
            
                face3 = theFace/255
                faceL.append(cv2.resize(face3, (224, 224)))
            

        print(len(faceL))
        text1, text2 = deepfakespredict(faceL)
        return text1 
model = tf.keras.models.load_model("FINAL-EFFICIENTNETV2-B0")
def deepfakespredict(Faces):

    
  
    total = 0
    real = 0
    fake = 0
    print(len(Faces))
    for face in Faces:

        face2 = face/255

      
        pred = model.predict(np.expand_dims(face, axis=0))[0]
        total+=1

        pred2 = pred[1]

        if pred2 > 0.5:
            fake+=1
        else:
            real+=1

    fake_ratio = fake/total

    text =""
    text2 = "Deepfakes Confidence: " + str(fake_ratio*100) + "%"

    if fake_ratio >= 0.5:
        text = "The video is FAKE."
    else:
        text = "The video is REAL."

    
    return text, text2
if __name__ == '__main__':
    app.run()

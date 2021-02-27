# utility file for emotion recognition from realtime webcam feed
import cv2
import sys
from tensorflow.keras.models import load_model
import time
import numpy as np
from decimal import Decimal
from model_utils import define_model, model_weights
from tensorflow.keras.preprocessing import image 

# loads and resizes an image
def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (48, 48))
    return True

# runs the realtime emotion detection 
def realtime_emotions():
    # load keras model
    model = define_model()
    model = model_weights(model)
    print('Model loaded')

    # for knowing whether prediction has started or not
    once = True
    # load haar cascade for face
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # list of given emotions
    EMOTIONS = ['anger', 'happy', 'neutral', 'sad']

    # store the emoji coreesponding to different emotions
    emoji_faces = []
    for index, emotion in enumerate(EMOTIONS):
        emoji_faces.append(cv2.imread('emojis/' + emotion.lower()  + '.png', -1))

    # set video capture device , webcam in this case
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)  # WIDTH
    video_capture.set(4, 480)  # HEIGHT

    # save current time
    prev_time = time.time()

    countt = 0
    # start webcam feed
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find face in the frame
        faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:

            # draw a rectangle bounding the face
            cv2.rectangle(frame, (x-10, y-70),
                            (x+w+20, y+h+40), (15, 175, 61), 4)

            curr_time = time.time()    
            if curr_time - prev_time >= 1:
                roi_gray=gray[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image  
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = np.reshape(roi_gray, (1, 48, 48, 1)) 
                mean = np.mean(roi_gray)
                std  = np.std(roi_gray)
                roi_gray = (roi_gray-mean)/(std+1e-7) 

				# do prediction
                result = model.predict(roi_gray)
                print(EMOTIONS[np.argmax(result[0])])
                prev_time = time.time()



            total_sum = np.sum(result[0])
            # select the emoji face with highest confidence
            emoji_face = emoji_faces[np.argmax(result[0])]
            for index, emotion in enumerate(EMOTIONS):
                text = str(
                    round(Decimal(result[0][index]/total_sum*100), 2) ) + "%"
                # for drawing progress bar
                cv2.rectangle(frame, (100, index * 20 + 10), (100 +int(result[0][index] * 100), (index + 1) * 20 + 4),
                                (255, 0, 0), -1)
                # for putting emotion labels
                cv2.putText(frame, emotion, (10, index * 20 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                # for putting percentage confidence
                cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    
                # overlay emoji on the frame for all the channels
                for c in range(0, 3):
                    # for doing overlay we need to assign weights to both foreground and background
                    foreground = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0)
                    background = frame[350:470, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                    frame[350:470, 10:130, c] = foreground + background
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

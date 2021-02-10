from model_utils import define_model, model_weights
import cv2
import os.path
import numpy as np
from tensorflow.keras.preprocessing import image 

# make prediction on image saved on disk
def prediction_path(path):
    # load keras model
    model = define_model()
    model = model_weights(model)
    
    # list of given emotions
    EMOTIONS = ['anger', 'happy', 'neutral', 'sad']

    if os.path.exists(path):
        # read the image
        img = cv2.imread(path, 0)
        # check if image is valid or not
        if img is None:
            print('Invalid image !!')
            return 
    else:
        print('Image not found')
        return

    # resize image for the model
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, (1, 48, 48, 1))
    # do prediction
    result = model.predict(img)
    print(result)
    print('Detected emotion: ' + str(EMOTIONS[np.argmax(result[0])]))
    
    return
def test(path):
    model = define_model()
    model = model_weights(model)
    images = cv2.imread(path)
    gray_img = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_detected = face_haar_cascade.detectMultiScale(
                                                gray_img,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE
                                            ) 
    print(faces_detected) 
    print(type(faces_detected)) 
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(images,(x,y),(x+w,y+h),(255,0,0))  
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  

        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1)) 
        mean = np.mean(roi_gray)
        std  = np.std(roi_gray)
        roi_gray = (roi_gray-mean)/(std+1e-7) 

    
        predictions = model.predict(roi_gray)  
        print(predictions)
        #find max indexed array  
        max_index = np.argmax(predictions[0])
        emotions = ('anger', 'happy', 'neutral', 'sad')
        predicted_emotion = emotions[max_index]  
        cv2.putText(images, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
    
    #resized_img = cv2.resize(images, (int(images.shape[0]/2), int(images.shape[1]/2)))  
    cv2.imshow("result", images)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()

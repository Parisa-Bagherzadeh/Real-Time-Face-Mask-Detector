from tensorflow.keras.models import load_model
import cv2
import numpy as np



file_face = 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(file_face)

model=load_model('FaceMaskDetection.h5')


video=cv2.VideoCapture(0)

frame_width=int(video.get(3))
frame_height=int(video.get(4))
size=(frame_width,frame_height)
result_frame=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'),20.0,size)

while(True):
    ret,frame=video.read()
    image=frame
    if ret==False:
        break

    #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = face_detector.detectMultiScale(frame, 1.3)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (224, 224))
        face_resize=face_resize/255
        face_resize=face_resize.reshape(1,224,224,3)
        pred=model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        result=np.argmax(pred)
        
        if result==0:
            cv2.putText(frame,'MASK',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),5,cv2.LINE_AA)
        elif result==1:
            cv2.putText(frame,'NO MASK',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),5,cv2.LINE_AA)  
    #farme=cv2.resize(frame,size)
    result_frame.write(frame)
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
video.release()
cv2.destroyAllWindows()   



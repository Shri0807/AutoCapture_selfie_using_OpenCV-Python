import cv2
import numpy as np

video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('//home/shri0807/Documents/Project/AutoCapture_Selfie_Smile/resources/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('/home/shri0807/Documents/Project/AutoCapture_Selfie_Smile/resources/haarcascade_smile.xml')

while True:
    success, img = video.read()
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayimg, 1.1, 4)
    cnt = 1
    keypressed = cv2.waitKey(1)

    for x, y, w, h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        smiles = smileCascade.detectMultiScale(grayimg, 1.8, 15)
        for x, y, w, h in smiles:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),2)
            print("Image "+str(cnt)+"Saved")
            path = r'/home/shri0807/Documents/Project/AutoCapture_Selfie_Smile/Output/img'+ str(cnt) + '.jpg'
            cv2.imwrite(path,img)
            cnt = cnt + 1
            if cnt >=2:
                break
    
    cv2.imshow('live video',img)
    if keypressed & 0xFF==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
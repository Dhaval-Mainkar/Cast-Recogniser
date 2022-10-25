from sre_constants import SUCCESS
import cv2
import numpy as np
import face_recognition
import os
import time

pTime=0

#for images stored in cast folder
path="Cast"
images=[]
classnames=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
  cur_img=cv2.imread(f'{path}/{cl}')
  images.append(cur_img)
  classnames.append(os.path.splitext(cl)[0])
print(classnames)

def findEncodings(images): #takes encoding from cast images
  encodelist=[]
  for img in images:
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    encode=face_recognition.face_encodings(img)[0]
    encodelist.append(encode)
  return encodelist

encodelistknown=findEncodings(images)
print(len(encodelistknown)) #number of images from the provided path

#video
cap=cv2.VideoCapture("everywhere.mp4")

while(cap.isOpened()):
    success,img=cap.read()

    imgs=cv2.resize(img,(0,0),None,0.25,0.25) #size of the window
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    facecurframe=face_recognition.face_locations(imgs)
    encodecurframe=face_recognition.face_encodings(imgs,facecurframe)

    for encodeface,faceloc in zip(encodecurframe,facecurframe):
            matches=face_recognition.compare_faces(encodelistknown,encodeface)
            facedist=face_recognition.face_distance(encodelistknown,encodeface) #this uses svm regressor to get the distance between the orginal image and the one displayed now
                                                                                #minimum the distance represents faces are similar
            print(facedist)
            matchindex=np.argmin(facedist)

            if matches[matchindex]:
                    name=classnames[matchindex].upper()
                    print(name)
                    y1,x2,y2,x1=faceloc
                    y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4 
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    #fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    #fps text line
    cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3) 

    cv2.imshow("video",img)

    if cv2.waitKey(10) & 0xFF==ord("q"): #when q is pressed it will end the video
        break

cap.release()
cv2.destroyAllWindows()
import cv2 as cv
import time
import math
import numpy as np
import HandTracking as ht
from pycaw.pycaw import AudioUtilities
cap= cv.VideoCapture(0)
wCam, hCam= 640,480
cap.set(3,wCam)
cap.set(4,hCam)
num_frames = 0
fps=0
vol=0
volPer = 0
volBar=400

start_time= time.time()
detector = ht.HandTracker()

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
minVol=volume.GetVolumeRange()[0]
maxVol=volume.GetVolumeRange()[1]

print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")

while True:
    success, img=cap.read()
    flipf= cv.flip(img,1)
    detector.detect_hands(flipf)
    lmList= detector.get_landmarks(flipf)
  
    if len(lmList)!=0:
        x1, y1 = lmList[4]['x'], lmList[4]['y']
        x2, y2 = lmList[8]['x'], lmList[8]['y']
        cx,cy= (x1+x2)//2, (y1+y2)//2

        cv.circle(flipf,(x1,y1),12,(255,0,255),cv.FILLED)
        cv.circle(flipf,(x2,y2),12,(255,0,255),cv.FILLED)
        cv.line(flipf,(x1,y1),(x2,y2),(255,0,255),2)
        cv.circle(flipf,(cx,cy), 7,(255,0,255), cv.FILLED)
        length=math.hypot(x2-x1, y2-y1)

        if length<50:
            cv.circle(flipf,(cx,cy), 13,(0,255,0), cv.FILLED)
        
        vol= np.interp(length,[50,200],[minVol,maxVol])
        volBar= np.interp(length,[50,200],[400,150])
        volPer = np.interp(length, [50, 200], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

    cv.rectangle(flipf, (50,150), (85,400), (0,255,0), 3)
    cv.rectangle(flipf, (50,int(volBar)), (85,400), (0,255,0), cv.FILLED)
    cv.putText(flipf,f'{int(volPer)}%',(40, 450), cv.FONT_HERSHEY_COMPLEX,1,
    (0,255,0),2)   
    num_frames += 1
    if num_frames >= 30:
        end_time = time.time()
        seconds = end_time - start_time
        fps = num_frames / seconds  
        num_frames = 0
        start_time = time.time()
    
    cv.putText(flipf,f"FPS:{int(fps)}",(30,50),
               cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    cv.imshow("Img",flipf)
    if cv.waitKey(1 )& 0xFF == ord('f'):
        break
cap.release()
cv.destroyAllWindows()

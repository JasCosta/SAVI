#!/usr/bin/env python3

import os
import cv2
from cv2 import imshow
import numpy as np
from PIL import Image
import pyttsx3
import time
engine = pyttsx3.init()
#--------
#Dataset
#--------
def dataset(id,name):
    # Load the cascade model for detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Capture the video from webcam
    cam = cv2.VideoCapture(0)
    
    # Count for save de picture
    count = 0
    
    names = ['None', 'Costa' ] 
    while(True):

        _, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/id." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            # Show  video to farmes dataset
            cv2.imshow('image to dataset', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 3: # Take 30 face sample and stop video
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
   
    # Release the VideoCapture object
    cam.release()
    cv2.destroyAllWindows()
    

   
#--------
#Train
#--------


    
    # Path for face image database
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier('/home/jc/Documents/savi_22-23-main/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_frontalface_default.xml');

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
if __name__ == "__main__":
   dataset()
#---------------------
# Bounding Box Class              
#----------------------

class BoundingBox:
    # Function  the Bounding Boxes
    def __init__(self, x1, y1, w, h):
        # Save local variables
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        # calculate area
        self.area = w * h
        # Calculates the other corner coordinates
        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h

    # Function  intersection of both bboxes
    def computeIOU(self, bbox2):
        # Gets the coordinates of the intersected rectangle
        x1_intr = min(self.x1, bbox2.x1)             
        y1_intr = min(self.y1, bbox2.y1)             
        x2_intr = max(self.x2, bbox2.x2)
        y2_intr = max(self.y2, bbox2.y2)
        # Gets the width height and area of the intersected rectangle
        w_intr = x2_intr - x1_intr
        h_intr = y2_intr - y1_intr
        A_intr = w_intr * h_intr
        # Calculates all the area of box boxes
        A_union = self.area + bbox2.area - A_intr
    
        # Returns the probability of being intersected
        return A_intr / A_union

    # Function Smallimage
    def extractSmallImage(self, image_full):
        self.extracted_face = image_full[self.y1:self.y2, self.x1:self.x2]


#----------------
# Detector Class                          
#-----------------

class Detection(BoundingBox):
    # Function  Detection
    def __init__(self, x1, y1, w, h, image_full, id):
        super().__init__(x1,y1,w,h)
        self.id = id
        self.extractSmallImage(image_full)
        # Initializes the variable that will tell if has a tracker associated
        self.assigned_to_tracker=False

    # Function that will draw the detection
    def draw(self, image_gui, color=(0,255,0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        names = ['None', 'Costa'] 
        
        # Recognition Model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        gray = cv2.cvtColor(image_gui,cv2.COLOR_BGR2GRAY)
   
        # Draws the rectangle around the detected part of the image
        image_gui = cv2.rectangle(image_gui,(self.x1,self.y1),(self.x2, self.y2),color,3)
        id, confidence = recognizer.predict(gray[self.y1:self.y2,self.x1:self.x2])
        if (confidence < 80):
           id = names[id]
           confidence = "  {0}%".format(round(100 - confidence))
           
           engine.say('Hello'+str(id))
           engine.runAndWait()
           time.sleep(4)
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            engine.say(" Who are you")
            engine.runAndWait()
            time.sleep(2)
        
        cv2.putText(image_gui, str(id), (self.x1+5,self.y1-5), font, 1, (255,255,255), 2)
        cv2.putText(image_gui, str(confidence), (self.x1+5,self.y2-5), font, 1, (255,255,0), 1) 
        return image_gui
    
#................
# Tracker Models                            
#................

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create() 
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create() 
if tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create() 
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create() 
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

#.................
# Tracker Class                            
#.................

class Tracker():
    # Function that initializes the Tracker
    def __init__(self, detection, id,image):
        # Creates an array to keep tracking of all the detections
        self.detections = [detection]
        # Creates an array of Bounding boxes to later draw them
        self.bboxes = []
        # Initializes the tracker model
        self.tracker = tracker
        # Gives an ID to the tracker
        self.id = id
        # Initializes the tracker and associates the detection
        self.addDetection(detection,image)


    # Function that will Draw the tracker based on the last bbox
    def draw(self, image_gui, color=(255,255,0)):
        # Gets the last Bounding Box to use its coordinates 
        bbox = self.bboxes[-1] # get last bbox
        # Draws the rectangle
        image_gui = cv2.rectangle(image_gui,(bbox.x1,bbox.y1),(bbox.x2, bbox.y2),color,3)
        # Puts the Tracking ID
        image_gui = cv2.putText(image_gui, 'T' + str(self.id), (bbox.x2-40, bbox.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # Returns the modified image
        return image_gui

    # Function that will add a Detection to the tracker so it can be later used to update itself
    def addDetection(self, detection,image):
        #Initializes the tracker
        self.tracker.init(image, (detection.x1, detection.y1, detection.w, detection.h))
        #Adds the last detection to the tracker
        self.detections.append(detection)
        #Sets the detection to have a tracker assigned
        detection.assigned_to_tracker = True
        bbox = BoundingBox(detection.x1, detection.y1, detection.w, detection.h)
        self.bboxes.append(bbox)

    # Function that will update the tracker if no detection is associated to the tracker
    def updateTracker(self,image_gray):
        # Calls the tracker model to update the tracer
         ret, bbox = self.tracker.update(image_gray)
         # Creates a new Bounding Box since the bbox given by the tracker as a different construction than what we use
         x1,y1,w,h = bbox
         bbox = BoundingBox(int(x1), int(y1), int(w), int(h))
         # Appends the bbox to be used in the Drawing
         self.bboxes.append(bbox)

    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text

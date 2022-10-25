#!/usr/bin/env python3

# Import all the libraries required
import cv2
import numpy as np
import os
from PIL import Image
import copy
from functions import Detection, Tracker, dataset


def main():
    # Load the cascade model for detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Recognition Model
    model = cv2.face.LBPHFaceRecognizer_create()

    # Count for save de picture
    count = 0
    # Create List emppty dor id
    face_id = {}
    # names related to ids: example ==> Costa: id=1,  etc
    names = ['None', ]

    image_rgb = cv2.VideoCapture(0)

    # For each person, enter one numeric face id
    id = (input('\n enter user id ==>  '))

    # Capture the video from webcam
    image_rgb = cv2.VideoCapture(0)

    name = (input('\n enter username and press enter=>  '))
    

    _ = dataset(int(id), name)

    names.append(name)
    print(names)
  
    
   # Initialize variables
    tracker_counter = 0
    trackers = []
    detection_counter = 0
    # Threshold for the relation of the detection and tracker
    iou_threshold = 0.8

    # Capture the video from webcam
    cap = cv2.VideoCapture(0)

    # Loops all the frames
    while True:
        # Read the frame
        ret, img = cap.read()
        # If the frame is invalid break the cycle
        if ret == False:
            break

        # Gets the timestamp
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create a copy of the image so we can do alterations to it and still preserve the original image
        image_gui = copy.deepcopy(img)

        # Create a list of detections and a counter that resets every cycle
        detections = []

        # Detect the faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4)
        print(f'Number of faces found = {len(faces)}')

        count = 0

        # Loops all the detected faces and Creates a detection and adds it to detection array
        for bbox in faces:
            x1, y1, w, h = bbox
            # Initializes the Detector
            detection = Detection(x1, y1, w, h, gray, id=detection_counter)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)
            #cv2.imshow('detection ' + str(detection.id), detection.image  )

        # ------------------------------------------
        # For each detection, see if there is a tracker to which it should be associated
        # ------------------------------------------

        # Loops all the detections and loops all the trackers and computes if they overlap and if they do add the new detection to the tracker
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                # Gets the last detection in the tracker to compute its overlap
                tracker_bbox = tracker.detections[-1]
                # Computes the overlap of both bboxes
                iou = detection.computeIOU(tracker_bbox)
                # If both bboxes overlap add the detection to the tracker
                if iou > iou_threshold: # associate detection with tracker
                    tracker.addDetection(detection, gray)

        # # ------------------------------------------
        # # Track using template matching
        # # ------------------------------------------

        # # Loops all the trackers and checks if any of the new detections is associated to the tracker if not update tracker
        # for tracker in trackers: # cycle all trackers
        #     # Gets the last detection ID in the tracker
        #     last_detection_id = tracker.detections[-1].id
        #     # Gets all the IDs of the Detections
        #     detection_ids = [d.id for d in detections]
        #     # If the last id in the tracker is not one of the new Detection update Tracker
        #     if not last_detection_id in detection_ids:
        #         # Update Tracker
        #         tracker.updateTracker(gray)

        # ------------------------------------------
        # Create Tracker for each detection
        # ------------------------------------------

        # Creates new trackers if the Detection has no tracker associated
        for detection in detections:
            # Checks to see if the Detections have a tracker associated to them
            if not detection.assigned_to_tracker:
                # Initializes the tracker
                tracker = Tracker(detection, id=tracker_counter, image=gray)
                # increment
                tracker_counter += 1
                # Deactivate Tracker if no detection for more than T
                trackers.append(tracker)

        # ------------------------------------------
        # Draw stuff
        # ------------------------------------------

        # Draw trackers
        for tracker in trackers:
            image_gui = tracker.draw(image_gui)

        # Draw all the detections
        for detection in detections:
            img = detection.draw(image_gui)

        # Display the results
        cv2.imshow('Window_name', img)

        # Stop if q key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
   # ------------
   # Termination
   # -------------

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

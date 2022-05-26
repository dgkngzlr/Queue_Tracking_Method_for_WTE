import cv2
import numpy as np


class Tracker:  #Tracks the bboxes with selected tracking algorithm
    
    def __init__(self, type_of_tracker):
        """
        Create object tracker.

        :param type_of_tracker string: OpenCV tracking algorithm
        """
        self.bboxes = []
        self.type = type_of_tracker
        self.tracker = self.generate_tracker(type_of_tracker)   
        self.multi_tracker = cv2.MultiTracker_create()  #cv2 object for multitracking
        self.sorted_bboxes=[]
        self.center_coord = 0

    def add_bb(self, frame, bboxes):       #initialize tracking for each bboxes

        for bbox in bboxes:
    
            self.multi_tracker.add(self.generate_tracker(self.type), frame, tuple(bbox))    
    
    def update(self, frame):    #updates the frame so that bboxes can move accordingly
        success, self.bboxes = self.multi_tracker.update(frame) #puts the new bboxes to the object attribute
    
    def restart(self):
        self.multi_tracker = cv2.MultiTracker_create() #Resets the object so that the empty bboxes dissappear
    
    def generate_tracker(self, type_of_tracker): #selecter for the desired tracking algorithm
        
        type_of_trackers = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        if type_of_tracker == type_of_trackers[0]:
            return cv2.TrackerBoosting_create()
        elif type_of_tracker == type_of_trackers[1]:
            return cv2.TrackerMIL_create()
        elif type_of_tracker == type_of_trackers[2]:
            return cv2.TrackerKCF_create()
        elif type_of_tracker == type_of_trackers[3]:
            return cv2.TrackerTLD_create()
        elif type_of_tracker == type_of_trackers[4]:
            return cv2.TrackerMedianFlow_create()
        elif type_of_tracker == type_of_trackers[5]:
            return cv2.TrackerGOTURN_create()
        elif type_of_tracker == type_of_trackers[6]:
            return cv2.TrackerMOSSE_create()
        elif type_of_tracker == type_of_trackers[7]:
            return cv2.TrackerCSRT_create()
        else:
            return None
            print('The name of the tracker is incorrect')

    def center_calc(self):
        self.sorter()
        if (len(self.sorted_bboxes)>0):
            self.center_coord = (self.sorted_bboxes[0]+ self.sorted_bboxes[2]/2)

    def sorter(self):
        closest= self.bboxes[0][0]
        for i in range(0,len(self.bboxes)-1):
            if (closest < self.bboxes[i][0]):
                closest= self.bboxes[i][0]
                self.sorted_bboxes=self.bboxes[i]
    
    def get_numpy_bboxes(self):
        return np.array(self.bboxes, dtype=np.int32)



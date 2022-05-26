import numpy as np
import time

from src.point import Dist

class Person:

    def __init__(self, bbox) -> None:
        
        self.x, self.y = int(bbox[0]), int(bbox[1])
        self.w, self.h = int(bbox[2]), int(bbox[3])
        self.center_x = int(bbox[0]) + int(bbox[2]) // 2
        self.center_y = int(bbox[1]) + int(bbox[3]) // 2
        self.center = [self.center_x, self.center_y]
        self.place = None

class Queue:
    
    def __init__(self) -> None:
        
        self.queue = []
        self.prev_queue = []
        self.n_entered_person = 0
        self.begin_time = time.time()
        self.avg_waiting_time = 0
        self.ref_line_points = []

    def sort(self, bboxes, to_right=True):

        pointfind = Dist()
        person_list = self._get_person_list(bboxes)

        if to_right:
            
            # All persons in the frame sorted by right to left 
            self.queue = sorted(person_list,
                                key=lambda person: person.center_x,
                                reverse=True)
            self.queue=pointfind.listcheck(self.queue,self.ref_line_points[0],self.ref_line_points[1])
            """
            * Get distance of people to the ref. line.
            * Ref. line points can be reached from self.ref_line_points
            * Update self.queue 
            """
        
        else:
            # All persons in the frame sorted by left to right 
            self.queue = sorted(person_list,
                                key=lambda person: person.center_x,
                                reverse=False)
            self.queue=pointfind.listcheck(self.queue,self.ref_line_points[0],self.ref_line_points[1])

            """
            * Get distance of people to the ref. line.
            * Ref. line points can be reached from self.ref_line_points
            * Update self.queue 
            """
        
        for i,person in enumerate(self.queue):

            person.place = i+1
        
    def _get_person_list(self, bboxes):

        person_list = []

        for bbox in bboxes:

            person = Person(bbox)
            person_list.append(person)
        
        return person_list
    
    def update(self):

        self.prev_queue.clear()
        self.prev_queue = self.queue.copy()
        self.queue.clear()

    def print_info(self):
        print("Present queue:", [i.center_x for i in self.queue])
        print("Previous queue:", [i.center_x for i in self.prev_queue])
    
    def get_sorted_bboxes(self):
        
        bboxes = []

        for person in self.queue:

            bboxes.append([person.x, person.y, person.w, person.h])
        
        return np.array(bboxes, dtype=np.int32)
    
    def is_eligable(self, idx, border, to_right=True):
        
        if len(self.prev_queue) != 0:

            if to_right:

                if self.prev_queue[idx].center_x < border and \
                            self.queue[idx].center_x >= border:
                            return True
            
            else:
                
                if self.prev_queue[idx].center_x > border and \
                            self.queue[idx].center_x <= border:
                            return True

        return False

    def count(self, border, to_right=True):
        
        if to_right:
                 
            if self.is_eligable(0, border):
                
                self.n_entered_person += 1
                self.avg_waiting_time = ((time.time() - self.begin_time) / \
                                        self.n_entered_person) * \
                                        len(self.queue)
            
            if len(self.queue) > 1:
                if self.is_eligable(1, border):
                    
                    self.n_entered_person += 1
                    self.avg_waiting_time = ((time.time() - self.begin_time) / \
                                            self.n_entered_person) * \
                                            len(self.queue)
        
        else :
               
            if self.is_eligable(0, border, to_right=False):
                
                self.n_entered_person += 1
                self.avg_waiting_time = ((time.time() - self.begin_time) // \
                                        self.n_entered_person) * \
                                        len(self.queue)

            if len(self.queue) > 1:
                if self.is_eligable(1, border, to_right=False):
                    
                    self.n_entered_person += 1
                    self.avg_waiting_time = ((time.time() - self.begin_time) // \
                                            self.n_entered_person) * \
                                            len(self.queue)
            
        
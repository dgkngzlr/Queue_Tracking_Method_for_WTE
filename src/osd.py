import cv2
import numpy as np
from src.entity import Queue

class OSD:
    
    def __init__(self, queue: Queue, border) -> None:
        
        self.frame = None
        self.queue = queue
        self.border = border
        self.color_list={"dodger_blue": (255, 144, 30),
                         "tomato": (71,99,255),
                         "medium_green": (154,250,0),
                         "dark_salmon": (122,150,233),
                         "pink": (222, 76, 212),
                         "galibarda": (255,0,255),
                         "orchid": (218,112,214),
                         "white": (255,255,255),
                         "red": (0,0,255)
                        }
        # For count click event
        self.i = 0
    
    def update_frame(self, frame):
        self.frame = frame
    
    def draw(self):
        
        alpha = 0.2
        bboxes = self.queue.get_sorted_bboxes()

        for i,bbox in enumerate(bboxes):

            x, y  = int(bbox[0]), int(bbox[1])
            w, h = int (bbox[2]), int(bbox[3])
            place_cord_x = (x + (w // 2)) - 5
            place_cord_y = (y + (h // 2)) - 10
            size = int(h*0.005 + w*0.005)
    
            if (x<=0):
                x=0

            if (y<=0):
                y=0

            # Draw bounding boxes
            overlay = self.frame.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), self.color_list["galibarda"], -1)
            self.frame = cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0)                                 # adds opacity
            self.frame = cv2.rectangle(self.frame, (x,y), ((x+w),(y+h)), \
                                       self.color_list["pink"], 1) #bbox outer frame
            
            # Put place number
            self.frame = cv2.putText(self.frame,
                                    f"{i+1}",
                                    (place_cord_x, place_cord_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    size,
                                    self.color_list["white"],
                                    2) #Total people on the screen

            # Draw line person to person            
            if i < len(bboxes) - 1: 

                self.frame = cv2.line(self.frame,
                                    tuple(self.queue.queue[i].center),
                                    tuple(self.queue.queue[i+1].center),
                                    self.color_list["medium_green"],
                                    2)
        # Total text
        self.frame = cv2.putText(self.frame,
                                f"Total : {len(self.queue.queue)}",
                                (5,15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                self.color_list["tomato"],
                                2)

        # Entered text
        self.frame = cv2.putText(self.frame,
                                f"Entered : {self.queue.n_entered_person}",
                                (5,30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                self.color_list["tomato"],
                                2)
        
        # Avg. Waiting Time text
        self.frame = cv2.putText(self.frame,
                                f"Avg. Waiting Time : {self.queue.avg_waiting_time:.2f} sec",
                                (5,45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                self.color_list["tomato"],
                                2)

        
        # Border line
        self.frame = cv2.line(self.frame,
                                    (self.border, 0),
                                    (self.border, self.frame.shape[1]),
                                    self.color_list["red"],
                                    2)
    
    def get_ref_line(self):

        cv2.imshow("frame", self.frame)
        cv2.setMouseCallback("frame", self._click_event)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass

        cv2.destroyAllWindows()
    
    
    def _click_event(self,event, x, y, flags, param):

        font = cv2.FONT_HERSHEY_SIMPLEX

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(self.queue.ref_line_points) == 2:
                cv2.destroyAllWindows()
                return None

            self.queue.ref_line_points.append([x,y])
            strXY = str(x)+", "+str(y)
            cv2.putText(self.frame, strXY, (x,y), font, 0.5, (255,255,0), 2)
            cv2.imshow("frame", self.frame)
            
            if len(self.queue.ref_line_points) == 2:
                cv2.line(self.frame, tuple(self.queue.ref_line_points[0]),
                         tuple(self.queue.ref_line_points[1]),
                        (255, 0, 0), thickness=2)
                cv2.imshow("frame", self.frame)
                
                

            


        

import cv2
import numpy as np
import time
from model.yolov4 import Yolov4
import conf.settings as cfg
from src.osd import OSD
from src.tracker import Tracker
from src.entity import Person, Queue


tracker = Tracker("MOSSE")
queue = Queue()
osd = OSD(queue, border=int(cfg.WIDTH*cfg.BORDER_SCALER))

model = Yolov4(cfg.WEIGHTS_PATH, cfg.CONFIG_PATH, cfg.LABELS_PATH)
model.load_yolo()

cap = cv2.VideoCapture(cfg.VIDEO_PATH)
ret, frame = cap.read()
frame = cv2.resize(frame, (cfg.WIDTH, cfg.HEIGHT), fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
_, detected_bboxes = model.predict(frame)
tracker.add_bb(frame, detected_bboxes)

osd.update_frame(frame)
osd.get_ref_line()

frame_counter = 0
sum_=0
c=1
while(True):
    begin_time = time.time()
    ret, frame = cap.read()

    try:

        frame = cv2.resize(frame, (cfg.WIDTH, cfg.HEIGHT), fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    except Exception as e:

        print("[ERR] Frame can not captured !")
        exit(-1)

    osd.update_frame(frame)

    if frame_counter % 30 == 0:
        tracker.restart()
        _, detected_bboxes = model.predict(frame)
        tracker.add_bb(frame, detected_bboxes)

    
    tracked_bboxes = tracker.get_numpy_bboxes()
    queue.sort(tracked_bboxes, to_right=cfg.TO_RIGHT)
    queue.count(border=int(frame.shape[1]*cfg.BORDER_SCALER), to_right=cfg.TO_RIGHT)
    
    # Update section
    osd.draw()
    tracker.update(frame)
    queue.update()
    
    cv2.imshow('frame', osd.frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
    delta = time.time() - begin_time
    sum_ += 1/delta
    
    print("AVG FPS: ",sum_ / c)

    frame_counter += 1
    c+=1
# After the loop release the cap objectq
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
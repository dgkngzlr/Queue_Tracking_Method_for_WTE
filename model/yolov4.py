import numpy as np
import cv2
import os
import time


class Yolov4:

    def __init__(self, WEIGHTS_PATH, CONFIG_PATH, LABELS_PATH):

        self.WEIGHTS_PATH = WEIGHTS_PATH
        self.CONFIG_PATH = CONFIG_PATH
        self.LABELS_PATH = LABELS_PATH
        
        self.LABELS = []
        self.USE_GPU = True

        # Load lables   
        try:
            with open(self.LABELS_PATH, "r") as f:

                for line in f.readlines():

                    self.LABELS.append(line.strip("\n"))
   
        except Exception as e:

            print("[LOAD LABELS ERROR]")

            exit(1)
    
    def load_yolo(self):
        
        print("[WAIT] Model is loading...")

        try :
            self.net = cv2.dnn.readNet(self.WEIGHTS_PATH, self.CONFIG_PATH)  # load YOLO algorithm.
        
        except Exception as e:

            print("[LOAD MODEL ERROR]")

            exit(1)

        if self.USE_GPU:

            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print('[INFO] Device : GPU')

        else:

            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print('[INFO] Device : CPU')
        
        # Get layers
        self.layer_names = self.net.getLayerNames()
        self.output_layers = []  


        for i in range(len(self.layer_names)):

            if i + 1 in self.net.getUnconnectedOutLayers():  # Output layerlarin indisini doner
                
                self.output_layers.append(self.layer_names[i])

        print("[DONE] Model is loaded")


    def __extract_boxes_confidences_classids(self,outputs, confidence, width, height):
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:            
                # Extract the scores, classid, and the confidence of the prediction
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]
                
                # Consider only the predictions that are above the confidence threshold
                if conf > confidence:
                    # Scale the bounding box back to the size of the image
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, w, h = box.astype('int')

                    # Use the center coordinates, width and height to get the coordinates of the top left corner
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))
                    classIDs.append(classID)

        return boxes, confidences, classIDs
        
    def predict(self, image,label="person", confidence=0.5, threshold=0.5):

        container = []
        bboxes = []

        # Get h x w
        height, width = image.shape[:2]
        
        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Forward pass
        outputs = self.net.forward(self.output_layers)

        # Extract bounding boxes, confidences and classIDs
        boxes, confidences, classIDs = self.__extract_boxes_confidences_classids(outputs, confidence, width, height)

        # Apply Non-Max Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        # Container = [ [ [bb_box] , confidence, label, center], .. ]
        # Create container 
        if len(idxs) > 0:
            
            for i in idxs.flatten():
                
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                center = (x + w // 2, y + h // 2)

                if self.LABELS[classIDs[i]] == label :
                    bboxes.append([x, y, w, h])
                    container.append([(x, y, w, h), confidences[i], self.LABELS[classIDs[i]], center])
        

        return container, np.array(bboxes, dtype=np.int32)

    # Draw bounding box
    def draw_bb(self, image, container, color = (0, 255, 0)):
        
        if len(container):
            
            for i in range(len(container)):
                
                x, y, w, h = container[i][0]

                cv2.rectangle(image, (x, y),
                              (x + w, y + h),
                              color,
                              1)

                text = "{}: {:.4f}".format(container[i][2], container[i][1])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image
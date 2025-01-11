from ultralytics import YOLO
import supervision as sv
import pickle
import os

'''
Tracking is asigning the same bounding box to the same object in different frames

We'll match the boundintg boxes to an ID

Smart tracking also tracks the trajectory of the object and uses color to help with this process

'''
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker  = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 20 #prevents memory issues

        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
           
        return detections

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)


        #for referencing
        tracks = {
            'players': [], # [{0: {1: {"bbox": [x1, y1, x2, y2]}}}, ... ] where each item is a frame
            'referees': [],
            'ball': [],
        }


        """
        The AI confuses the players with the goal keeper
        The following function aims to override player with goal keeper label
        """
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names 
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # Convert detections to superviosion detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            '''
            from supervision class_id array 2 correlates to player and 1 correlates to goalkeeper
            we want to convert the detection of goalkeeper to player
            
            '''
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']


            # Track the objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({}) # {0: {1: {"bbox": [x1, y1, x2, y2]}}}
            tracks["referees"].append({})
            tracks["ball"].append({})


            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]


                #Only one ball
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            return tracks
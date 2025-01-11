from ultralytics import YOLO
import supervision as sv

'''
Tracking is asigning the same bounding box to the same object in different frames

We'll match the boundintg boxes to an ID

Smart tracking also track the trajectory of the object and uses color to help with this process

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
            break #to not run the loop for all the frames
        return detections

    def get_object_tracks(self, frames):
        
        detections = self.detect_frames(frames)

        tracks = {
            "players":

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

            print(detection_with_tracks)
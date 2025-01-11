import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read() #ret is the flag that tells if the frame has ended 
        if not ret:
            break
        frames.append(frame)
    
    return frames

"""
    Save a list of video frames to a video file.

    Parameters:
    out_video_frames (list): List of frames (numpy arrays) to be written to the video file.
    out_video_path (str): Path to the output video file.

    The function uses OpenCV's VideoWriter to write the frames to a video file.

    cv2.VideoWriter Parameters:
    ---------------------------
    out_video_path (str): Path to the output video file.
    fourcc (int): 4-character code of codec used to compress the frames.
    fps (float): Frame rate of the created video stream.
    frameSize (tuple): Size of the video frames (width, height).

"""
def save_video(output_video_frames, output_video_path):

    fourcc = cv2.VideoWriter_fourcc(*'XVID') #format of the video
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (output_video_frames[0].shape[1],output_video_frames[0].shape[0])) #24.0 is the frames per second
    for frame in output_video_frames:
        out.write(frame)
    out.release()
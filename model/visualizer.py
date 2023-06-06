import cv2 as cv
from tqdm import tqdm


def broadcast_video(raw_video_path, frame_indexes, output_path, fps=None):
    raw_video = cv.VideoCapture(raw_video_path)
    width = int(raw_video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(raw_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  
    if fps == None:
        fps = int(raw_video.get(cv.CAP_PROP_FPS))
    
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    video = cv.VideoWriter(output_path, fourcc, float(fps), (width, height))
    current_frame = 0
    pbar = tqdm(total=len(frame_indexes))
    while True:
        ret, frame = raw_video.read()
        if not ret:
            break
        if current_frame in frame_indexes:
            video.write(frame)
            pbar.update(1)
        
        current_frame += 1
      
    raw_video.release()
    video.release()
    pbar.close()

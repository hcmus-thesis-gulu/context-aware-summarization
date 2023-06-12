import cv2 as cv
from tqdm import tqdm


def broadcast_video(input_video_path, frame_indices,
                    output_video_path, fragment_width,
                    fps=None):
    raw_video = cv.VideoCapture(input_video_path)
    width = int(raw_video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(raw_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  
    if fps == None:
        fps = int(raw_video.get(cv.CAP_PROP_FPS))
    
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    video = cv.VideoWriter(output_video_path, fourcc,
                           float(fps), (width, height))
    cur_idx = 0
    pbar = tqdm(total=len(frame_indices))
    kf_idx = 0
    
    while True:
        ret, frame = raw_video.read()
        if not ret:
            break
        
        while kf_idx < len(frame_indices) and frame_indices[kf_idx] < cur_idx - fragment_width:
            kf_idx += 1
        if kf_idx < len(frame_indices) and abs(frame_indices[kf_idx] - cur_idx) <= fragment_width:
            video.write(frame)
        
        if cur_idx in frame_indices:
            pbar.update(1)
        
        cur_idx += 1
      
    raw_video.release()
    video.release()
    pbar.close()

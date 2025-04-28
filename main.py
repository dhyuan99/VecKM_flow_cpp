import numpy as np
from VecKM_flow import SliceNormalFlowEstimator
import matplotlib.pyplot as plt

import os
os.environ["LD_LIBRARY_PATH"] = "."

dt, k = 24, 8
model_name = f"640x480_{dt}ms_C64_k{k}"
estimator = SliceNormalFlowEstimator(model_name, 500000, 640, 480, 64, k)

os.system(f"mkdir -p tmp_frames/")

events_t = np.load(f"demo_data/dataset_events_t.npy")
events_xy = np.load(f"demo_data/dataset_events_xy.npy")

intervals = np.arange(events_t[0], events_t[-1], dt / 1000)
intervals = np.concatenate((intervals, [events_t[-1]]))
for i in range(len(intervals)-1):
    indices = np.where((events_t >= intervals[i]) & (events_t < intervals[i+1]))[0]
    if len(indices) == 0:
        continue
    
    events_xy_i = events_xy[indices]
    events_t_i  = events_t[indices]

    events = np.concatenate((events_t_i[:, None], events_xy_i), axis=1)
    events = np.ascontiguousarray(events, dtype=np.float32)
    target_indices = np.arange(0, len(events))
    
    flow = estimator.predict_flows(
        events, events.shape[0], 
        target_indices, target_indices.shape[0], 
        events_t_i[0], dt/2000)
    
    if len(events) > 5000:
        r = np.random.choice(np.arange(len(events)), 5000, replace=False)
    else:
        r = np.arange(len(events))
        
    plt.figure(figsize=(16,12))
    plt.scatter(events[:, 1], events[:, 2], c='gray', s=1)
    plt.quiver(
        events[r, 1], events[r, 2], 
        flow[r, 0], -flow[r, 1], color='b', scale_units='xy', scale=1
    )
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"tmp_frames/{str(i).zfill(4)}.png")
    plt.close()
    
import os
import cv2

def images_to_video(image_folder, output_video, fps):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
images_to_video(f"tmp_frames", f"demo.mp4", 10)
os.system(f"rm -rf tmp_frames/")
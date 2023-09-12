"""
    mail@kaiploeger.net
"""

import numpy as np
import cv2
from pathlib import Path


def main():
    h,w,c = 1080, 1920, 3
    frame_rate = 30
    duration = 3

    frames = [np.random.randint(0, 255, (h,w,c), dtype=np.uint8) for _ in range(frame_rate * duration)]

    # write frames to avi
    file_name = Path(__file__).parent / 'test_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(file_name), fourcc, frame_rate, (w, h))  # !!! (w, h) have to be in reverse order compared to frame array !!!
    for frame in frames:
        out.write(frame)
    out.release()

    # write frames to mp4
    file_name = Path(__file__).parent / 'test_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(file_name), fourcc, frame_rate, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

if __name__ == '__main__':
    main()




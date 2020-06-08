# -*- coding: utf-8 -*-
"""
Created on ~

@author: Gerardo Cervantes
"""

import cv2


# video_path is a path to the video file
# output_image_dir is the path to the directory where it will save all the images
# num_frames_to_skip means it should skip this many frames until saving another frame, if 1 then will save every frame
def save_video_frames(video_path, output_image_dir):
    vidcap = cv2.VideoCapture(video_path)

    if not vidcap:
        return False
    has_next_frame = True
    i = 0
    while has_next_frame:
        has_next_frame, image = vidcap.read()
        frame_output_path = output_image_dir + '/' + str(i) + '.png'
        cv2.imwrite(frame_output_path, image)  # Save image
        i += 1
    return True


if __name__ == "__main__":

    video_path = r'D:\mario_data\mario-cam.mp4'
    image_output_dir = r'D:\mario_data\mario-cam'
    save_video_frames(video_path, image_output_dir)




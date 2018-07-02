""" Data utility functions """
from __future__ import division

import os
import cv2
import numpy as np
import time
import sys

def construct_filenames_and_labels(data_dir, dic):
    filenames = []
    labels = []
    for label_name in os.listdir(data_dir):
        lab_dir = os.path.join(data_dir, label_name)
        for video_dir in os.listdir(lab_dir):
            vid_dir = os.path.join(lab_dir, video_dir)
            for f in os.listdir(vid_dir):
                filenames.append(os.path.join(vid_dir, f))
                labels.append(int(dic[label_name]) - 1)

    return filenames, labels

def construct_optical_flow_filenames(filenames, volume_depth, mode):
    start = time.time()
    flow_filenames = []
    flow_parameters = []

    for filename in filenames:
        if (mode == "test"):
            filename = filename.replace("_sampled", "")
        dirs = filename.split("/")

        dirs[2] = "{}_flow".format(dirs[2])
        frame_id = int(os.path.splitext(dirs[-1])[0])
        frame_ids_init = list(range(frame_id+1,frame_id+volume_depth+1))
        frame_ids = frame_ids_init.copy()

        for id in reversed(frame_ids_init):
            flow_x = "{}_flow_x.jpg".format(id)
            filename = os.path.join(os.path.join(*(dirs[:-1])), flow_x)
            if (os.path.exists(filename)):
                break;
            frame_ids = [id - 1 for id in frame_ids]

        frame_ids = [str(id) for id in frame_ids]
        frame_ids = ["+"] + frame_ids
        flow_filenames.append(os.path.join(os.path.join(*(dirs[:-1])), "-".join(frame_ids)))

    end = time.time()
    print(end-start)
    print(sys.getsizeof(flow_filenames))
    print(np.array(flow_filenames).nbytes)

    #return flow_filenames, flow_parameters
    return flow_filenames

def sample_test(test_data_dir_sampled, test_data_dir, n_sample_frames):
    # Create the directory for the sampled frames
    os.makedirs(test_data_dir_sampled)
    for label_name in os.listdir(test_data_dir):
        lab_dir = os.path.join(test_data_dir, label_name)
        lab_dir_sampled = os.path.join(test_data_dir_sampled, label_name)
        os.makedirs(lab_dir_sampled)
        for video_dir in os.listdir(lab_dir):
            vid_dir = os.path.join(lab_dir, video_dir)
            vid_dir_sampled = os.path.join(lab_dir_sampled, video_dir)
            os.makedirs(vid_dir_sampled)
            video_frames = os.listdir(vid_dir)

            # Sample n_sample_frames frames from the video
            n_frames = len(video_frames) # number of frames in the video
            interval = n_frames // (n_sample_frames - 1)
            start = (n_frames % (n_sample_frames - 1)) // 2 + 1 # Starting index
            for i in range(n_sample_frames):
                if (start + i * interval > n_frames):
                    frame_ind = n_frames
                else:
                    frame_ind = start + i * interval
                filename = "{}.jpg".format(frame_ind)
                file_path = os.path.join(vid_dir, filename)
                image = cv2.imread(file_path)
                cv2.imwrite(os.path.join(vid_dir_sampled, filename), img=image)

""" Data utility functions """
from __future__ import division

import os
import cv2
import numpy as np
import time
import sys

IMG_DIM = 224
def create_and_save_samples(filename, dir, dir_sampled):
    file_path = os.path.join(dir, filename)
    image = cv2.imread(file_path)
    base = os.path.splitext(os.path.basename(filename))[0]
    # cv2.imshow("A", image)
    # cv2.waitKey()
    # print(file_path)
    # print(image.shape)

    crop_top_left = image[:IMG_DIM, :IMG_DIM]
    crop_top_right = image[:IMG_DIM, -IMG_DIM:]
    crop_bot_left = image[-IMG_DIM:, :IMG_DIM]
    crop_bot_right = image[-IMG_DIM:, -IMG_DIM:]
    img_shapes = (image.shape[0], image.shape[1])
    x = (img_shapes[0] - 224) // 2
    y = (img_shapes[1] - 224) // 2
    crop_centre = image[x:x+IMG_DIM, y:y+IMG_DIM]

    # Flip horizontally
    crop_top_left_flip = cv2.flip(crop_top_left, 1)
    crop_top_right_flip = cv2.flip(crop_top_right, 1)
    crop_bot_left_flip = cv2.flip(crop_bot_left, 1)
    crop_bot_right_flip = cv2.flip(crop_bot_right, 1)
    crop_centre_flip = cv2.flip(crop_centre, 1)

    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_top_left.jpg".format(base)), img=crop_top_left)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_top_right.jpg".format(base)), img=crop_top_right)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_bot_left.jpg".format(base)), img=crop_bot_left)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_bot_right.jpg".format(base)), img=crop_bot_right)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_centre.jpg".format(base)), img=crop_centre)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_top_left_flip.jpg".format(base)), img=crop_top_left_flip)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_top_right_flip.jpg".format(base)), img=crop_top_right_flip)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_bot_left_flip.jpg".format(base)), img=crop_bot_left_flip)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_bot_right_flip.jpg".format(base)), img=crop_bot_right_flip)
    cv2.imwrite(os.path.join(dir_sampled, "{}_crop_centre_flip.jpg".format(base)), img=crop_centre_flip)

    # return (crop_top_left, crop_top_right, crop_bot_left, crop_bot_right, crop_centre,
    #         crop_top_left_flip, crop_top_right_flip, crop_bot_left_flip,
    #         crop_bot_right_flip, crop_centre_flip)

def construct_filenames_and_labels(data_dir, dic):
    start = time.time()
    filenames = []
    labels = []
    for label_name in os.listdir(data_dir):
        lab_dir = os.path.join(data_dir, label_name)
        for video_dir in os.listdir(lab_dir):
            vid_dir = os.path.join(lab_dir, video_dir)
            for f in os.listdir(vid_dir):
                filenames.append(os.path.join(vid_dir, f))
                labels.append(int(dic[label_name]) - 1)
    end = time.time()
    print(end-start)
    print(sys.getsizeof(filenames))
    print(np.array(filenames).nbytes)

    return filenames, labels

def construct_optical_flow_filenames(filenames, volume_depth):
    start = time.time()
    flow_filenames = []
    flow_parameters = []
    #count = 0
    for filename in filenames:
        dirs = filename.split("/")

        dirs[2] = "{}_flow".format(dirs[2])
        base_name = os.path.splitext(dirs[-1])[0]
        split = base_name.split("_")
        print(split)
        frame_id = int(split[-1])
        frame_ids_init = list(range(frame_id+1,frame_id+volume_depth+1))
        frame_ids = frame_ids_init.copy()

        for id in reversed(frame_ids_init):
            split[-1] = str(id)
            flow_x = "{}_flow_x.jpg".format("_".join(split))
            filename = os.path.join(os.path.join(*(dirs[:-1])), flow_x)
            if (os.path.exists(filename)):
                break;
            frame_ids = [id - 1 for id in frame_ids]

        flow_filename = []
        flow_param = []
        for id in frame_ids:
            split[-1] = str(id)
            flow_x = "{}_flow_x.jpg".format("_".join(split))
            flow_y = "{}_flow_y.jpg".format("_".join(split))
            flow_params = "{}.npy".format("_".join(split))
            filename_x = os.path.join(os.path.join(*(dirs[:-1])), flow_x)
            filename_y = os.path.join(os.path.join(*(dirs[:-1])), flow_y)
            filename_params = os.path.join(os.path.join(*(dirs[:-1])), flow_params)
            #flow_param.append(np.load(filename_params))
            flow_filename.append((filename_x, filename_y))
        flow_filenames.append(flow_filename)
        #flow_parameters.append(flow_param)
        #if count % 10000 == 0:
        #    print(filename)
        #count += 1
    end = time.time()
    print(end-start)
    print(sys.getsizeof(flow_filenames))
    print(np.array(flow_filenames).nbytes)

    #return flow_filenames, flow_parameters
    return flow_filenames

def test_create_and_save_sample():
    filename = "v_ApplyEyeMakeup_g08_c01_45.jpg"
    dir = "/home/hayk/workspace/ISTC_Grant/data/UCF-101_test01/ApplyEyeMakeup/g08_c01"
    dir_sampled =  "/home/hayk/workspace/ISTC_Grant/data/UCF-101_test01_sampled/ApplyEyeMakeup/g08_c01"

    cropped_images = create_and_save_sample(filename, dir, dir_sampled)
    cv2.imshow("top left", cropped_images[0])
    cv2.waitKey()
    cv2.imshow("top right", cropped_images[1])
    cv2.waitKey()
    cv2.imshow("bot left", cropped_images[2])
    cv2.waitKey()
    cv2.imshow("bot_right", cropped_images[3])
    cv2.waitKey()
    cv2.imshow("centre", cropped_images[4])
    cv2.waitKey()
    cv2.imshow("top left flipped", cropped_images[5])
    cv2.waitKey()
    cv2.imshow("top right flipped", cropped_images[6])
    cv2.waitKey()
    cv2.imshow("bot left flipped", cropped_images[7])
    cv2.waitKey()
    cv2.imshow("bot_right flipped", cropped_images[8])
    cv2.waitKey()
    cv2.imshow("centre flipped", cropped_images[9])
    cv2.waitKey()

    for i, _ in enumerate(cropped_images):
        assert cropped_images[i].shape[0] == IMG_DIM
        assert cropped_images[i].shape[1] == IMG_DIM

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
            base_ind = video_frames[0].rindex("_") # Find the index of the base name end
            interval = n_frames // (n_sample_frames - 1)
            start = (n_frames % (n_sample_frames - 1)) // 2 + 1 # Starting index
            base_name = video_frames[0][:base_ind+1] # base name of the frames
            for i in range(n_sample_frames):
                if (start + i * interval > n_frames):
                    frame_ind = n_frames
                else:
                    frame_ind = start + i * interval
                filename = "{}{}.jpg".format(base_name, frame_ind)
                create_and_save_samples(filename, vid_dir, vid_dir_sampled)

if __name__ == "__main__":
    test_create_and_save_sample()

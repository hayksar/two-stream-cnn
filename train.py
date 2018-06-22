"""Train the model"""
from collections import Counter
import argparse
import logging
import os
import random
import numpy as np

import tensorflow as tf

from model.input_fn import input_spatial_fn, input_temporal_fn, input_two_stream_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import spatial_model_fn, temporal_model_fn
from model.training import train_and_evaluate
from data_utils import sample_test, construct_filenames_and_labels, construct_optical_flow_filenames


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/toy_model_temporal',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='../data',
                    help="Directory containing the dataset")
parser.add_argument('--dataset', default='UCF-101',
                    help="Dataset name"),
parser.add_argument('--class_ind', default='../data/ucfTrainTestlist/classInd.txt',
                    help="Labels file name"),
parser.add_argument('--split', default='1',
                    help="Split number to use for training and testing")
parser.add_argument('--stream', default='temporal',
                    help="spatial, temporal all two_stream")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # number of frames to be sampled from the test videos
    n_sample_frames = 25

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    # overwritting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    set_name = args.dataset
    n_split = args.split
    if (n_split != "all"):
        train_data_dir = os.path.join(data_dir, "{}_train0{}".format(set_name, n_split))
    else:
        train_data_dir = os.path.join(data_dir, "{}_train".format(set_name))
    if (n_split != "all"):
        test_data_dir = os.path.join(data_dir, "{}_test0{}".format(set_name, n_split))
        test_data_dir_sampled = os.path.join(data_dir, "{}_test0{}_sampled".format(set_name, n_split))
    else:
        test_data_dir = os.path.join(data_dir, "{}_test".format(set_name))
        test_data_dir_sampled = os.path.join(data_dir, "{}_test_sampled".format(set_name))

    # Construct the dictionary of labels from classInd.txt
    label_dic = {}
    with open(args.class_ind) as f:
        for line in f:
            (val, key) = line.split()
            label_dic[key] = int(val)

    # Get the filenames and labels from the train set
    train_filenames, train_labels = construct_filenames_and_labels(train_data_dir, label_dic)
    assert len(train_filenames) == len(train_labels)
    # Specify the size of the dataset we train on
    params.train_size = len(train_filenames)
    if args.stream != "spatial":
        #train_flow_filenames, train_flow_params = construct_optical_flow_filenames(train_filenames, params.volume_depth)
        train_flow_filenames = construct_optical_flow_filenames(train_filenames, params.volume_depth)
    # Create the iterator over the train dataset
    if args.stream == "spatial":
        train_inputs = input_spatial_fn(True, train_filenames, train_labels, params)
        # Free up the memory
        del train_filenames
        del train_labels
    elif args.stream == "temporal":
        #train_inputs = input_temporal_fn(True, train_flow_filenames, train_labels, train_flow_params, params)
        train_inputs = input_temporal_fn(True, train_flow_filenames, train_labels, params)
        # Free up the memory
        del train_filenames
        del train_flow_filenames
        del train_labels
    else:
        train_inputs = input_two_stream_fn(True, train_filenames, train_flow_filenames, train_labels, params)
        # Free up the memory
        del train_filenames
        del train_flow_filenames
        del train_labels

    # Get the filenames and labels from the test set
    # NOTE: During the test time we evaluate the accuracy on the video file

    # Create the iterators over the test dataset
    if args.stream == "spatial":
        if not os.path.exists(test_data_dir_sampled):
            sample_test(test_data_dir_sampled, test_data_dir, n_sample_frames=n_sample_frames)
        test_filenames, test_labels = construct_filenames_and_labels(test_data_dir_sampled, label_dic)
        assert len(test_filenames) == len(test_labels)
        # Specify the size of the dataset we evaluate on
        params.test_size = len(test_filenames)
        test_inputs = input_spatial_fn(False, test_filenames, test_labels, params)
        del test_filenames
        del test_labels
    elif args.stream == "temporal":
        test_filenames, test_labels = construct_filenames_and_labels(test_data_dir, label_dic)
        assert len(test_filenames) == len(test_labels)
        # Specify the size of the dataset we evaluate on
        params.test_size = len(test_filenames)
        test_flow_filenames = construct_optical_flow_filenames(test_filenames, params.volume_depth)
        test_inputs = input_temporal_fn(False, test_flow_filenames, test_labels, params)
        # Free up the memory
        del test_flow_filenames
        del test_filenames
        del test_labels
    else:
        test_inputs = input_two_stream_fn(False, test_filenames, test_flow_filenames, test_labels, params)
        # Free up the memory
        del test_flow_filenames
        del test_filenames
        del test_labels


    # Define the model
    logging.info("Creating the model...")
    if (args.stream == "spatial"):
        train_model_spec = spatial_model_fn('train', train_inputs, params)
        test_model_spec = spatial_model_fn('test', test_inputs, params, reuse=True)
    elif (args.stream == "temporal"):
        train_model_spec = temporal_model_fn('train', train_inputs, params)
        test_model_spec = temporal_model_fn('test', test_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, test_model_spec, args.model_dir, params, args.restore_from)

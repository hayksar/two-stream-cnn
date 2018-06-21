"""Create the input data pipeline using `tf.data`"""

import os
import numpy as np
import random
import tensorflow as tf

def convert_to_original_range(flow_x, flow_y, params):
    flow = tf.concat([flow_x, flow_y], axis=2)
    print(flow.get_shape().as_list())
    #print(params[100].get_shape().as_list())
    flow = tf.cast(flow, tf.float32) * params[:,-1] / 255.0 + params[:,0]
    print(flow.get_shape().as_list())

    return flow



def _parse_spatial_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    # Randomly crop an image with size*size
    image = tf.random_crop(value=image, size=[size, size, 3])

    return image, label

def get_flow_images(flow_filenames_params):
    #flow_filenames = flow_filenames_params[0]
    flow_filenames = flow_filenames_params
    #flow_param = flow_filenames_params[1]
    flow_x_string = tf.read_file(flow_filenames[0])
    flow_y_string = tf.read_file(flow_filenames[1])
    flow_x_decoded = tf.image.decode_jpeg(flow_x_string, channels=1)
    flow_y_decoded = tf.image.decode_jpeg(flow_y_string, channels=1)
    #print(flow_x_decoded.get_shape().as_list())
    #flow = convert_to_original_range(flow_x_decoded, flow_y_decoded, flow_param)
    flow = tf.concat([flow_x_decoded, flow_y_decoded], axis=2)

    return flow



def _parse_temporal_function(flow_volume_filenames, label, size, volume_depth):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """

    get_flows_fn = lambda flow_filenames_params: get_flow_images(flow_filenames_params)
    #flows = tf.map_fn(get_flows_fn, (flow_volume_filenames, flow_param), dtype = tf.float32)
    #flows = tf.map_fn(get_flows_fn, flow_volume_filenames, dtype = tf.float32)
    flows = tf.map_fn(get_flows_fn, flow_volume_filenames, dtype = tf.uint8)
    print(flows.get_shape().as_list())
    #flow = tf.concat(flows, axis=2)
    flow = tf.reshape(flows, [tf.shape(flows)[1], tf.shape(flows)[2], -1])
    print(flow.get_shape().as_list())
    flow = tf.image.convert_image_dtype(flow, tf.float32)
    flow = tf.random_crop(value=flow, size=[size, size, 2*volume_depth])
    print(size)

    return flow, label


def train_spatial_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """

    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def input_spatial_fn(is_training, filenames, labels, params):
    """Input function for the spatial stream.

    The filenames have format "v_{label_name}_{video_id}_{clip_id}_{frame_number}.jpg".
    For instance: "data_dir/v_ApplyEyeMakeup_g08_c01_45.jpg".

    Arguments
    ---------
    is_training :   (bool) whether to use the train or test pipeline.
                    At training, we shuffle the data and have multiple epochs
    filenames   :   (list) filenames of the images (frames)
    labels      :   (list) corresponding list of labels
    params      :   (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    parse_fn = lambda f, l: _parse_spatial_function(f, l, params.image_size)
    train_fn = lambda s, l: train_spatial_preprocess(s, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.test_batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs

def input_temporal_fn(is_training, flow_filenames, labels, params, flow_params=None):
    """Input function for the temporal stream.

    The filenames have format "v_{label_name}_{video_id}_{clip_id}_{frame_number}.jpg".
    For instance: "data_dir/v_ApplyEyeMakeup_g08_c01_45.jpg".

    Arguments
    ---------
    is_training :   (bool) whether to use the train or test pipeline.
                    At training, we shuffle the data and have multiple epochs
    filenames   :   (list) filenames of the images (frames)
    labels      :   (list) corresponding list of labels
    params      :   (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(flow_filenames)
    assert len(flow_filenames) == len(labels), "Filenames and labels should have same length"
    #assert len(flow_filenames) == len(flow_params), "Filenames and labels should have same length"
    flow_params = np.array(flow_params)

    # Create a Dataset serving batches of images and labels
    #parse_fn = lambda f, l, p: _parse_temporal_function(f, l, p, params.image_size, params.volume_depth)
    parse_fn = lambda f, l: _parse_temporal_function(f, l, params.image_size, params.volume_depth)

    if is_training:
        #dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(flow_filenames), tf.constant(labels), tf.constant(flow_params)))
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(flow_filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        #dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(flow_filenames), tf.constant(labels), tf.constant(flow_params)))
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(flow_filenames), tf.constant(labels)))
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.test_batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    flows, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': flows, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs

def input_two_stream_fn(is_training, filenames, labels, flow_filenames, params, stream):
    """Input function for the UCF-101 dataset.

    The filenames have format "v_{label_name}_{video_id}_{clip_id}_{frame_number}.jpg".
    For instance: "data_dir/v_ApplyEyeMakeup_g08_c01_45.jpg".

    Arguments
    ---------
    is_training :   (bool) whether to use the train or test pipeline.
                    At training, we shuffle the data and have multiple epochs
    filenames   :   (list) filenames of the images (frames)
    labels      :   (list) corresponding list of labels
    params      :   (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    stream      :   (string) "spatial", "temporal" or "two_stream"
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"
    assert len(flow_filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size, params.volume_depth, stream)
    if stream == "two_stream":
        train_fn = lambda s, t, l: train_preprocess(s, t, l, params.use_random_flip, stream)
    elif stream == "temporal":
        train_fn = lambda t, l: train_preprocess(None, t, l, params.use_random_flip, stream)
    else:
        train_fn = lambda s, l: train_preprocess(s, None, l, params.use_random_flip, stream)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels), tf.constant(flow_filenames)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.test_batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    if stream == "two_stream":
        images, flows, labels = iterator.get_next()
    if stream == "temporal":
        flows, labels = iterator.get_next()
    else:
        images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    if stream == "two_stream":
        inputs = {'images': images, 'labels': labels, 'flows': flows, 'iterator_init_op': iterator_init_op}
    if stream == "temporal":
        inputs = {'flows': flows, 'labels': labels, 'iterator_init_op': iterator_init_op}
    else:
        inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs

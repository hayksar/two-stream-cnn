"""Define the model."""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


# def build_spatial_model(is_training, inputs, params):
#     """Compute logits of the model (output distribution)
#
#     Arguments
#     ---------
#     is_training :   (bool) whether we are training or not
#     inputs      :   (dict) contains the inputs of the graph (features, labels...)
#                         this can be `tf.placeholder` or outputs of `tf.data`
#     params      :   (Params) hyperparameters
#
#     Returns
#     -------
#     output      :   (tf.Tensor) output of the model
#     """
#     images = inputs['images']
#
#     assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]
#
#     out = images
#     # Define the number of channels of each convolution
#     # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
#     num_channels = params.num_channels
#     bn_momentum = params.bn_momentum
#     channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8, num_channels * 16]
#     for i, c in enumerate(channels):
#         with tf.variable_scope('block_{}'.format(i+1)):
#             out = tf.layers.conv2d(out, c, 3, padding='same')
#             if params.use_batch_norm:
#                 out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
#             out = tf.nn.relu(out)
#             out = tf.layers.max_pooling2d(out, 2, 2)
#
#     assert out.get_shape().as_list() == [None, 7, 7, num_channels * 16]
#
#     out = tf.reshape(out, [-1, 7 * 7 * num_channels * 16])
#     with tf.variable_scope('fc_1'):
#         out = tf.layers.dense(out, num_channels * 16)
#         if params.use_batch_norm:
#             out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
#         out = tf.nn.relu(out)
#     with tf.variable_scope('fc_2'):
#         logits = tf.layers.dense(out, params.num_labels)
#
#     return logits

def build_spatial_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Arguments
    ---------
    is_training :   (bool) whether we are training or not
    inputs      :   (dict) contains the inputs of the graph (features, labels...)
                        this can be `tf.placeholder` or outputs of `tf.data`
    params      :   (Params) hyperparameters

    Returns
    -------
    output      :   (tf.Tensor) output of the model
    """
    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

    # vgg = nets.vgg
    # with slim.arg_scope(vgg.vgg_arg_scope()):
    #     logits, _ = vgg.vgg_16(images, num_classes=params.num_labels, is_training=is_training)

    resnet = nets.resnet_v1
    if (is_training):
        with slim.arg_scope(resnet.resnet_arg_scope()):
            logits, _ = resnet.resnet_v1_50(images, num_classes=params.num_labels, is_training=is_training)
    else:
        with slim.arg_scope(resnet.resnet_arg_scope()):
            logits, _ = resnet.resnet_v1_50(images, num_classes=params.num_labels, reuse=True, is_training=is_training)

    return tf.squeeze(logits, [1, 2], name='SpatialSqueeze'), resnet


def build_temporal_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Arguments
    ---------
    is_training :   (bool) whether we are training or not
    inputs      :   (dict) contains the inputs of the graph (features, labels...)
                        this can be `tf.placeholder` or outputs of `tf.data`
    params      :   (Params) hyperparameters

    Returns
    -------
    output      :   (tf.Tensor) output of the model
    """
    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, params.volume_depth*2]

    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8, num_channels * 16]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.get_shape().as_list() == [None, 7, 7, num_channels * 16]

    out = tf.reshape(out, [-1, 7 * 7 * num_channels * 16])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, num_channels * 16)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(out, params.num_labels)

    return logits


def model_fn(mode, inputs, params, stream, reuse=False):
    """Model function defining the graph operations.

    Arguments
    ---------
    mode        :   (string) can be 'train' or 'eval'
    inputs      :   (dict) contains the inputs of the graph (features, labels...)
                        this can be `tf.placeholder` or outputs of `tf.data`
    params      :   (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    reuse       :   (bool) whether to reuse the weights

    Returns
    -------
    model_spec  :   (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)
    # logits, vgg = build_spatial_model(is_training, inputs, params)
    # # Restore only the layers up to fc7 (included)
    # # Calling function `init_fn(sess)` will load all the pretrained weights.
    # model_path = "/Users/hayk/workspace/ISTC_Grant/remote/two_stream/model/vgg_16.ckpt"
    # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
    # print(variables_to_restore)
    # print("AAAAAAAA")
    # init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)
    # #init_fn = tf.train.Saver(variables_to_restore)
    #
    # # Initialization operation from scratch for the new "fc8" layers
    # # `get_variables` will only return the variables whose name starts with the given pattern
    # fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
    # print(fc8_variables)
    # fc8_init = tf.variables_initializer(fc8_variables)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    #with tf.variable_scope('model', reuse=reuse):
    # Compute the output distribution of the model and the predictions
    if stream == "spatial":
        logits, resnet = build_spatial_model(is_training, inputs, params)
    elif stream == "temporal":
        logits = build_temporal_model(is_training, inputs, params)
    elif stream == "two_stream":
        logits_spatial = build_spatial_model(is_training, inputs, params)
        logits_temporal = build_temporal_model(is_training, inputs, params)
    if is_training:
        predictions = tf.argmax(logits, 1)
    else:
        logits_mean = tf.reduce_mean(logits, axis=0)
        predictions = tf.argmax(logits_mean)
    # Restore only the layers up to fc7 (included)
    # Calling function `init_fn(sess)` will load all the pretrained weights.
    # model_path = "/Users/hayk/workspace/ISTC_Grant/remote/two_stream/model/vgg_16.ckpt"
    # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['model/vgg_16/fc8'])
    model_path = "model/resnet_v1_50.ckpt"
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['resnet_v1_50/logits'])
    #init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)
    init_fn=tf.train.Saver(variables_to_restore)

    # Initialization operation from scratch for the new "fc8" layers
    # `get_variables` will only return the variables whose name starts with the given pattern
    #fc8_variables = tf.contrib.framework.get_variables('model/vgg_16/fc8')
    fc8_variables = tf.contrib.framework.get_variables('resnet_v1_50/logits')
    fc8_init = tf.variables_initializer(fc8_variables)

    # Define loss and accuracy
    if is_training:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    else:
        loss = 0.0
    if is_training:
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
    else:
        accuracy = tf.cast(tf.equal(labels[0], predictions), tf.float32)

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step, var_list=fc8_variables)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step, var_list=fc8_variables)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        if is_training:
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
                'loss': tf.metrics.mean(loss)
                }
        else:
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels[0], predictions=predictions),
                'loss': tf.metrics.mean(loss)
                }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    #tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    # for label in range(0, params.num_labels):
    #     mask_label = tf.logical_and(mask, tf.equal(predictions, label))
    #     incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
    #     tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    #model_spec['labels'] = labels
    #model_spec['images'] = inputs['images']
    model_spec['init_fn'] = init_fn
    model_spec['fc8_init'] = fc8_init

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec

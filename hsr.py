# The architecture is inspired by LeNet-5 (LeCun, 1998)
import os

import tensorflow as tf

# Parameter
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
BATCH_SIZE = 5
NUM_EPOCHS = 2
NUM_CLASS = 5
NUM_CHANNELS = 3
CONV1_FILTER_SIZE  = 32
CONV1_FILTER_COUNT  = 4
CONV2_FILTER_SIZE  = 16
CONV2_FILTER_COUNT  = 6
HIDDEN_LAYER_SIZE = 400

def read_images(data_dir):
    pattern = os.path.join(data_dir, '*.png')
    filenames = tf.train.match_filenames_once(pattern, name='list_files')
    
    queue = tf.train.string_input_producer(
        filenames, 
        num_epochs=NUM_EPOCHS, 
        shuffle=True, 
        name='queue')
    
    reader = tf.WholeFileReader()
    filename, content = reader.read(queue, name='read_image')
    filename = tf.Print(
        filename, 
        data=[filename],
        message='loading: ')
    filename_split = tf.string_split([filename], delimiter='/')
    label_id = tf.string_to_number(tf.substr(filename_split.values[1], 
        0, 1), out_type=tf.int32)
    label = tf.one_hot(
        label_id-1, 
        5, 
        on_value=1.0, 
        off_value=0.0, 
        dtype=tf.float32)

    img_tensor = tf.image.decode_png(
        content, 
        dtype=tf.uint8, 
        channels=3,
        name='img_decode')

    # Preprocess the image, Performs random transformations
    # Random flip
    img_tensor_flip = tf.image.random_flip_left_right(img_tensor)

    # Random brightness
    img_tensor_bri = tf.image.random_brightness(img_tensor_flip, 
        max_delta=0.2)

    # Per-image scaling
    img_tensor_std = tf.image.per_image_standardization(img_tensor_bri)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    example_batch, label_batch = tf.train.shuffle_batch(
        [img_tensor_std, label], 
        batch_size=BATCH_SIZE,
        shapes=[(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), (NUM_CLASS)],
        capacity=capacity, 
        min_after_dequeue=min_after_dequeue,
        name='train_shuffle')

    return example_batch, label_batch

# `images` is a 4-D tensor with the shape:
# [n_batch, img_height, img_width, n_channel]
def inference(images):
    # Convolutional layer 1
    with tf.name_scope('conv1'):
        W = tf.Variable(
            tf.truncated_normal(
                shape=(
                    CONV1_FILTER_SIZE, 
                    CONV1_FILTER_SIZE,
                    NUM_CHANNELS,
                    CONV1_FILTER_COUNT),
                dtype=tf.float32,
                stddev=5e-2), 
            name='weights')
        b = tf.Variable(
            tf.zeros(
                shape=(CONV1_FILTER_COUNT), 
                dtype=tf.float32), 
            name='biases')
        conv = tf.nn.conv2d(
            input=images, 
            filter=W,
            strides=(1, 1, 1, 1), 
            padding='SAME',
            name='convolutional')
        conv_bias = tf.nn.bias_add(conv, b)
        conv_act = tf.nn.relu(
            features=conv_bias, 
            name='activation')
        pool1 = tf.nn.max_pool(
            value=conv_act, 
            ksize=(1, 2, 2, 1), 
            strides=(1, 2, 2, 1), 
            padding='SAME', 
            name='subsampling')

    # Convolutional layer 2
    with tf.name_scope('conv2'):
        W = tf.Variable(
            tf.truncated_normal(
                shape=(
                    CONV2_FILTER_SIZE, 
                    CONV2_FILTER_SIZE,
                    CONV1_FILTER_COUNT,
                    CONV2_FILTER_COUNT),
                dtype=tf.float32,
                stddev=5e-2), 
            name='weights')
        b = tf.Variable(
            tf.zeros(
                shape=(CONV2_FILTER_COUNT), 
                dtype=tf.float32), 
            name='biases')
        conv = tf.nn.conv2d(
            input=pool1, 
            filter=W,
            strides=(1, 1, 1, 1), 
            padding='SAME',
            name='convolutional')
        conv_bias = tf.nn.bias_add(conv, b)
        conv_act = tf.nn.relu(
            features=conv_bias, 
            name='activation')
        pool2 = tf.nn.max_pool(
            value=conv_act, 
            ksize=(1, 2, 2, 1), 
            strides=(1, 2, 2, 1), 
            padding='SAME', 
            name='subsampling')

    # Hidden layer
    with tf.name_scope('hidden'):
        conv_output_size = 28800 
        W = tf.Variable(
            tf.truncated_normal(
                shape=(conv_output_size, HIDDEN_LAYER_SIZE), 
                dtype=tf.float32,
                stddev=5e-2), 
            name='weights')
        b = tf.Variable(
            tf.zeros(
                shape=(HIDDEN_LAYER_SIZE), 
                dtype=tf.float32), 
            name='biases')
        reshape = tf.reshape(
            tensor=pool2, 
            shape=[BATCH_SIZE, -1])
        h1 = tf.nn.relu(
            features=tf.add(tf.matmul(reshape, W), b),
            name='activation')

    # Softmax layer
    with tf.name_scope('softmax'):
        W = tf.Variable(
            tf.truncated_normal(
                shape=(HIDDEN_LAYER_SIZE, NUM_CLASS), 
                dtype=tf.float32,
                stddev=5e-2),
            name='weights')
        b = tf.Variable(
            tf.zeros(
                shape=(NUM_CLASS), 
                dtype=tf.float32), 
            name='biases')
        logits = tf.add(tf.matmul(h1, W), b, name='logits')

    return logits

def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
        labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def training(loss, learning_rate=5e-3):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    predictions = tf.argmax(logits, 1, name='predictions')
    correct_predictions = tf.equal(predictions, 
        tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 
        'float'), name='accuracy')
    return accuracy


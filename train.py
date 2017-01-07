import tensorflow as tf
import hsr
import sys
import os
import time

if __name__ == '__main__':
    # % python train.py folder_name
    if len(sys.argv) < 2:
        print 'Usage: python', sys.argv[0], 'training_data/'
        sys.exit(1)

    data_dir = sys.argv[1]

    image_total = 0
    for subdir, dirs, files in os.walk(data_dir):
        for file_name in files:
            if file_name.split('.')[-1] == 'png':
                image_total += 1

    checkpoint_dir = os.path.abspath('checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create graph
    images, labels = hsr.read_images(data_dir)
    logits = hsr.inference(images)
    loss = hsr.loss(logits, labels)
    train = hsr.training(loss, learning_rate=5e-2)
    accuracy = hsr.evaluation(logits, labels)

    # Run the graph
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    try:
        batch_i = 1
        total_batch = 0
        epoch = 1
        start_time = time.time()
        while not coord.should_stop():
            loss_value, acc_value, _ = session.run([
                loss, 
                accuracy, 
                train])
            elapsed_time = time.time() - start_time
            print 'epoch:', epoch, 'batch:', batch_i, 'loss:', loss_value, 'accuracy:', acc_value, 'duration: %.3fs' % elapsed_time
            batch_i += 1
            total_batch += hsr.BATCH_SIZE
            if total_batch >= image_total:
                epoch += 1
                total_batch = 0
                batch_i = 1

            saver.save(session, checkpoint_prefix)
            start_time = time.time()

    except tf.errors.OutOfRangeError:
        print ''
        print 'Done.'
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()

    coord.join(threads)

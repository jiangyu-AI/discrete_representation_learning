#  Neural Discrete Representation Learning repl."
#  Dec. 2017
"""training script for Neural Discrete Representation Learning repl."""

import sys
import os
import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from six.moves import xrange

from model import VQVAE
from pixelcnn import PixelCNN

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument(
    '--data_set',
    type=str,
    default='mnist',
    help='Dataset to use, the other one is cifar10')

parser.add_argument(
    '--decay_steps',
    type=int,
    default=20000,
    help='Number of images to process in a batch')

parser.add_argument(
    '--decay_val',
    type=float,
    default=1.0,
    help='Number of images to process in a batch')

parser.add_argument(
    '--decay_staircase',
    type=bool,
    default=False,
    help='Number of images to process in a batch')

parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Number of images to process in a batch')

parser.add_argument(
    '--data_dir',
    type=str,
    default='./tmp/mnist_data',
    help='Path to directory containing the MNIST dataset')

parser.add_argument(
    '--model_dir',
    type=str,
    default='./tmp/last.ckpt',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of epochs to train.')

parser.add_argument(
    '--data_format',
    type=str,
    default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')

parser.add_argument(
    '--log_dir',
    type=str,
    default='./tmp',
    help='Data set i.e. input data to train on')

parser.add_argument(
    '--train_num',
    type=int,
    default=60000,
    help='200000 for cifar10, 60000 for mnist, num of trianing examples in one epoch')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.0002,
    help='Data set i.e. input data to train on')


parser.add_argument(
    '--beta',
    type=float,
    default=0.25,
    help='Time interval to summary')

parser.add_argument(
    '--K',
    type=int,
    default=4,
    help='4 for mnist, 10 for cifar10. Num of embedding vectors')

parser.add_argument(
    '--D',
    type=int,
    default=128,
    help='128 for mnist, 256 for cifar10. Embedding vector dimension')

parser.add_argument(
    '--grad_clip',
    type=float,
    default=1.0,
    help='Please set 1.0 for mnist, 5.0 for cifar10')

parser.add_argument(
    '--num_layers',
    type=int,
    default=12,
    help='Time interval to summary')

parser.add_argument(
    '--num_feature_maps',
    type=int,
    default=32,
    help='32 for mnist, 64 for cifar10')

parser.add_argument(
    '--summary_interval',
    type=int,
    default=1000,
    help='Time interval to summary')

parser.add_argument(
    '--save_interval',
    type=int,
    default=10000,
    help='Time interval to save model')

parser.add_argument(
    '--random_seed',
    type=int,
    default=1,
    help='Graph level random seed')


# The codes from
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
DATA_DIR = 'datasets/cifar10'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def maybe_download_and_extract():
    import sys, tarfile
    from six.moves import urllib
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(DATA_DIR)


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    record_bytes = 1 + 32*32*3

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [1]), tf.int32)
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [1],
                         [1 + 32*32*3]),
        [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def get_image(train=True,num_epochs=None):
    maybe_download_and_extract()
    if train:
        filenames = [os.path.join(DATA_DIR, 'cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    else:
        filenames = [os.path.join(DATA_DIR, 'cifar-10-batches-bin', 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs)
    read_input = read_cifar10(filename_queue)
    return tf.cast(read_input.uint8image, tf.float32) / 255.0, tf.reshape(read_input.label,[])


def train_embedding( data_set,
                     random_seed,
                     log_dir,
                     train_num,
                     batch_size,
                     learning_rate,
                     decay_val,
                     decay_steps,
                     decay_staircase,
                     beta,
                     K,
                     D,
                     save_interval,
                     summary_interval):
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # get input data
    if data_set == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("datasets/mnist", one_hot=False)
        x = tf.placeholder(tf.float32,[None,784])
        input_data = tf.image.resize_images(
            tf.reshape(x,[-1,28,28,1]),
            (24,24),
            method=tf.image.ResizeMethod.BILINEAR)
    elif data_set == 'cifar10':
        image,_ = get_image() # 32*32*3
        input_data = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=4,
            capacity=batch_size*10,
            min_after_dequeue=batch_size*2)

    # model
    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = learning_rate
        with tf.variable_scope('params') as params:
            pass
        net = VQVAE(learning_rate,global_step,beta,input_data,K,D,data_set,params,True)

    # summary
    with tf.variable_scope('misc'):
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('recon',net.recon_loss)
        tf.summary.scalar('vq',net.embed_loss)
        tf.summary.scalar('commit',beta*net.commit_loss)
        tf.summary.scalar('nll',tf.reduce_mean(net.nll))
        tf.summary.image('origin',input_data,max_outputs=4)
        tf.summary.image('recon',net.reconstruction,max_outputs=4)
        summary_op = tf.summary.merge_all()
        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(log_dir,sess.graph)

    # train
    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(train_num),dynamic_ncols=True):
            if data_set == 'mnist':
                batch_xs, _= mnist.train.next_batch(batch_size)
                it,loss,_ = sess.run([global_step,net.loss,net.train_op],feed_dict={x:batch_xs})
            elif data_set == 'cifar10':
                it,loss,_ = sess.run([global_step,net.loss,net.train_op])
            if( it % save_interval == 0 ):
                net.save(sess,log_dir,step=it)
            if( it % summary_interval == 0 ):
                tqdm.write('[%5d] Loss: %1.3f'%(it,loss))
                if data_set == 'mnist':
                    summary = sess.run(summary_op,feed_dict={x:batch_xs})
                elif data_set == 'cifar10':
                    summary = sess.run(summary_op)
                summary_writer.add_summary(summary,it)
    except Exception as e:
        coord.request_stop(e)
    finally:
        net.save(sess,log_dir)
        coord.request_stop()
        coord.join(threads)


def transform_data(data_set,
                   model_dir,
                   batch_size,
                   beta,
                   K,
                   D):
    # get input data
    if data_set == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("datasets/mnist", one_hot=False)
        x = tf.placeholder(tf.float32,[None,784])
        input_data = tf.image.resize_images(
            tf.reshape(x,[-1,28,28,1]),
            (24,24),
            method=tf.image.ResizeMethod.BILINEAR)
    elif data_set == 'cifar10':
        image,label = get_image(num_epochs=1) # 32*32*3
        images,labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=batch_size,
            allow_smaller_final_batch=True)

    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        if data_set == 'mnist':
            net = VQVAE(None,None,beta,input_data,K,D,data_set,params,False)
        elif data_set == 'cifar10':
            x_ph = tf.placeholder(tf.float32,[None,32,32,3])
            net= VQVAE(None,None,beta,x_ph,K,D,data_set,params,False)


    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    net.load(sess,model_dir)

    if data_set == 'mnist':
        xs,ys = mnist.train.images, mnist.train.labels
        ks = []
        for i in tqdm(range(0,len(xs),batch_size)):
            batch = xs[i:i+batch_size]
            k = sess.run(net.k,feed_dict={x:batch})
            ks.append(k)
        ks = np.concatenate(ks,axis=0)
        np.savez(os.path.join(os.path.dirname(model_dir),'ks_ys.npz'),ks=ks,ys=ys) # [3*3] indices of latent represetations embeddings and corresponding labels
    elif data_set == 'cifar10':
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        try:
            ks = []
            ys = []
            while not coord.should_stop():
                x,y = sess.run([images,labels])
                k = sess.run(net.k,feed_dict={x_ph:x})
                ks.append(k)
                ys.append(y)
                print('.', end='', flush=True)
        except tf.errors.OutOfRangeError:
            print('Extracting Finished')
        ks = np.concatenate(ks,axis=0)
        ys = np.concatenate(ys,axis=0)
        np.savez(os.path.join(os.path.dirname(model_dir),'ks_ys.npz'),ks=ks,ys=ys)
        coord.request_stop()
        coord.join(threads)


class Latent_data():
    def __init__(self,path,validation_size=1):
    #def __init__(self,path,validation_size=5000):
        from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
        from tensorflow.contrib.learn.python.learn.datasets import base

        data = np.load(path)
        train = DataSet(data['ks'][validation_size:], data['ys'][validation_size:],reshape=False,dtype=np.uint8,one_hot=False) 
        validation = DataSet(data['ks'][:validation_size], data['ys'][:validation_size],reshape=False,dtype=np.uint8,one_hot=False)
        self.size = data['ks'].shape[1]
        self.data = base.Datasets(train=train, validation=validation, test=None)


def train_prior(data_set,
                random_seed,
                model_dir,
                train_num,
                batch_size,
                learning_rate,
                decay_val,
                decay_steps,
                decay_staircase,
                grad_clip,
                K,
                D,
                beta,
                num_layers,
                num_feature_maps,
                summary_interval,
                save_interval):
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    log_dir = os.path.join(os.path.dirname(model_dir),'pixelcnn')
    if data_set == 'mnist':
        input_height = 24
        input_width = 24
        input_channel = 1
    elif data_set == 'cifar10':
        input_height = 32
        input_width = 32
        input_channel =3
    latent_data = Latent_data(os.path.join(os.path.dirname(model_dir),'ks_ys.npz'))

    # model_dir for Generate Images
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        _not_used = tf.placeholder(tf.float32,[None,input_height,input_width,input_channel])# 32*32*3,24*24*1
        vq_net = VQVAE(None,None,beta,_not_used,K,D,data_set,params,False)

    # model_dir for Training Prior
    with tf.variable_scope('pixelcnn'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = learning_rate
        net = PixelCNN(learning_rate,global_step,grad_clip,latent_data.size,vq_net.embeds,K,D,10,num_layers,num_feature_maps)
    with tf.variable_scope('misc'):
        tf.summary.scalar('loss',net.loss)
        summary_op = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sample_images = tf.placeholder(tf.float32,[None,input_height,input_width,input_channel])#24*24*1,32*32*3
        sample_summary_op = tf.summary.image('samples',sample_images,max_outputs=20)

    # train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    vq_net.load(sess,model_dir)
    summary_writer = tf.summary.FileWriter(log_dir,sess.graph)
    for step in tqdm(xrange(train_num),dynamic_ncols=True):
        batch_xs, batch_ys = latent_data.data.train.next_batch(batch_size)
        it,loss,_ = sess.run([global_step,net.loss,net.train_op],feed_dict={net.X:batch_xs,net.h:batch_ys})
        if( it % save_interval == 0 ):
            net.save(sess,log_dir,step=it)
        if( it % summary_interval == 0 ):
            tqdm.write('[%5d] Loss: %1.3f'%(it,loss))
            summary = sess.run(summary_op,feed_dict={net.X:batch_xs,net.h:batch_ys})
            summary_writer.add_summary(summary,it)
        if( it % (summary_interval * 2) == 0 ):
            sampled_zs,log_probs = net.sample_from_prior(sess,np.arange(10),2)
            sampled_ims = sess.run(vq_net.gen,feed_dict={vq_net.latent:sampled_zs})
            summary_writer.add_summary(
                sess.run(sample_summary_op,feed_dict={sample_images:sampled_ims}),it)
    net.save(sess,log_dir)




def main(unused_argv):

    train_embedding( data_set = FLAGS.data_set, 
                     random_seed = FLAGS.random_seed,
                     log_dir = FLAGS.log_dir,
                     train_num = FLAGS.train_num,
                     batch_size = FLAGS.batch_size,
                     learning_rate = FLAGS.learning_rate,
                     decay_val = FLAGS.decay_val,
                     decay_steps = FLAGS.decay_steps,
                     decay_staircase = FLAGS.decay_staircase,
                     beta = FLAGS.beta,
                     K = FLAGS.K,
                     D = FLAGS.D,
                     save_interval = FLAGS.save_interval,
                     summary_interval = FLAGS.summary_interval )
                    
    transform_data(data_set = FLAGS.data_set,
                   model_dir = FLAGS.model_dir,
                   batch_size = FLAGS.batch_size,
                   beta = FLAGS.beta,
                   K = FLAGS.K,
                   D = FLAGS.D)

    tf.reset_default_graph() 

    train_prior(FLAGS.data_set,
                FLAGS.random_seed,
                FLAGS.model_dir,
                FLAGS.train_num,
                FLAGS.batch_size,
                FLAGS.learning_rate,
                FLAGS.decay_val,
                FLAGS.decay_steps,
                FLAGS.decay_staircase,
                FLAGS.grad_clip,
                FLAGS.K,
                FLAGS.D,
                FLAGS.beta,
                FLAGS.num_layers,
                FLAGS.num_feature_maps,
                FLAGS.summary_interval,
                FLAGS.save_interval)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


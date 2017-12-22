'''VQVAE Model'''

import sys
import tensorflow as tf
import numpy as np
from ops import *

class VQVAE(object):
    def __init__(self,lr,global_step,beta,x,K,D,data_set,param_scope,is_training=False,optimizer=tf.train.AdamOptimizer(0.0002)):
        with tf.variable_scope(param_scope):
            if data_set == 'mnist':
                with tf.variable_scope('enc') as enc_param_scope :
                    enc_spec = [
                        Conv2d('conv2d_1',1,D//4,data_format='NHWC'), #d//4=16
                        lambda t,**kwargs : tf.nn.relu(t),
                        Conv2d('conv2d_2',D//4,D//2,data_format='NHWC'), # 
                        lambda t,**kwargs : tf.nn.relu(t),
                        Conv2d('conv2d_3',D//2,D,data_format='NHWC'), #d=64
                        lambda t,**kwargs : tf.nn.relu(t),
                    ]
                with tf.variable_scope('dec') as dec_param_scope :
                    dec_spec = [
                        TransposedConv2d('tconv2d_1',D,D//2,data_format='NHWC'),
                        lambda t,**kwargs : tf.nn.relu(t),
                        TransposedConv2d('tconv2d_2',D//2,D//4,data_format='NHWC'),
                        lambda t,**kwargs : tf.nn.relu(t),
                        TransposedConv2d('tconv2d_3',D//4,1,data_format='NHWC'),
                        lambda t,**kwargs : tf.nn.sigmoid(t),
                    ]
            elif data_set == 'cifar10':
                def _residual(t,conv3,conv1):
                    return conv1(tf.nn.relu(conv3(tf.nn.relu(t))))+t
                from functools import partial

                with tf.variable_scope('enc') as enc_param_scope :
                    enc_spec = [
                        Conv2d('conv2d_1',3,D,data_format='NHWC'),
                        lambda t,**kwargs : tf.nn.relu(t),
                        Conv2d('conv2d_2',D,D,data_format='NHWC'),
                        lambda t,**kwargs : tf.nn.relu(t),
                        partial(_residual,
                                conv3=Conv2d('res_1_3',D,D,3,3,1,1,data_format='NHWC'),
                                # kernel 3*3, stride 1*1
                                conv1=Conv2d('res_1_1',D,D,1,1,1,1,data_format='NHWC')),
                                # kernel 1*1, stride 1*1
                        partial(_residual,
                                conv3=Conv2d('res_2_3',D,D,3,3,1,1,data_format='NHWC'),
                                conv1=Conv2d('res_2_1',D,D,1,1,1,1,data_format='NHWC')),
                    ]
                with tf.variable_scope('dec') as dec_param_scope :
                    dec_spec = [
                        partial(_residual,
                                conv3=Conv2d('res_1_3',D,D,3,3,1,1,data_format='NHWC'),
                                conv1=Conv2d('res_1_1',D,D,1,1,1,1,data_format='NHWC')),
                        partial(_residual,
                                conv3=Conv2d('res_2_3',D,D,3,3,1,1,data_format='NHWC'),
                                conv1=Conv2d('res_2_1',D,D,1,1,1,1,data_format='NHWC')),
                        TransposedConv2d('tconv2d_1',D,D,data_format='NHWC'),
                        lambda t,**kwargs : tf.nn.relu(t),
                        TransposedConv2d('tconv2d_2',D,3,data_format='NHWC'),
                        lambda t,**kwargs : tf.nn.sigmoid(t),
                    ]
            with tf.variable_scope('embed') :
                embeds = tf.get_variable('embed', [K,D],initializer=tf.truncated_normal_initializer(stddev=0.02))
                self.embeds = embeds

        with tf.variable_scope('forward') as forward_scope:
            # Encoder Pass
            _t = x #[batch,x_h,x_w,input_channels] i.e. _*24*24*1 for MNIST, 128*32*32*3 for CIFAR10
            for block in enc_spec :
                _t = block(_t)
            z_e = _t # [batch,latent_h,latent_w,D] i.e. _*3*3*64 for MNIST, 128*8*8*256 for CIFAR10
            # Middle Area (Compression or Discretize   
            _t = tf.tile(tf.expand_dims(z_e,-2),[1,1,1,K,1]) #[batch,latent_h,latent_w,K,D]
            _e = tf.reshape(embeds,[1,1,1,K,D])
            _t = tf.norm(_t-_e,axis=-1)
            k = tf.argmin(_t,axis=-1) # -> [latent_h,latent_w]
            z_q = tf.gather(embeds,k)
            self.z_e = z_e # -> [batch,latent_h,latent_w,D]
            self.k = k # indices of embeds
            self.z_q = z_q # -> [batch,latent_h,latent_w,D] _*3*3*64 for MNIST, 128*8*8*256 for CIFAR10
            # Decoder Pass
            _t = z_q
            for block in dec_spec:
                _t = block(_t)
            self.reconstruction = _t # [batch, x_h, x_w, 1] i.e. _*24*24*1 for mnist, *3 for cifar10
            # Losses
            self.recon_loss = tf.reduce_mean((self.reconstruction - x)**2,axis=[0,1,2,3])
            self.embed_loss = tf.reduce_mean(tf.norm(tf.stop_gradient(self.z_e) - z_q,axis=-1)**2,axis=[0,1,2])
            self.commit_loss = tf.reduce_mean(tf.norm(self.z_e - tf.stop_gradient(z_q),axis=-1)**2,axis=[0,1,2])
            self.loss = self.recon_loss + self.embed_loss + beta * self.commit_loss
            self.nll = -1.*(tf.reduce_mean(tf.log(self.reconstruction),axis=[1,2,3]) + tf.log(1/tf.cast(K,tf.float32)))/tf.log(2.)
        if( is_training ):
            with tf.variable_scope('backward'): # Decoder, encoder, embed Grads
                decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,dec_param_scope.name)
                decoder_grads = list(zip(tf.gradients(self.loss,decoder_vars),decoder_vars))
                encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,enc_param_scope.name)
                grad_z = tf.gradients(self.recon_loss,z_q)
                encoder_grads = [(tf.gradients(z_e,var,grad_z)[0]+beta*tf.gradients(self.commit_loss,var)[0],var) for var in encoder_vars]
                embed_grads = list(zip(tf.gradients(self.embed_loss,embeds),[embeds]))
                optimizer = tf.train.AdamOptimizer(lr)
                self.train_op= optimizer.apply_gradients(decoder_grads+encoder_grads+embed_grads,global_step=global_step)
        else :# to decode latent vector  
            size = self.z_e.get_shape()[1]
            self.latent = tf.placeholder(tf.int64,[None,size,size])
            _t = tf.gather(embeds,self.latent)
            for block in dec_spec:
                _t = block(_t)
            self.gen = _t
        save_vars = {('train/'+'/'.join(var.name.split('/')[1:])).split(':')[0] : var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }
        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 3)

    def calc_total_loss(self, X):
        return self.sess.run(self.loss, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.k, feed_dict={self.x:X})

    def generate(self, latent_index):
        # decoding latent vectors
        z_q = tf.gather(self.embeds, self.latent_index)
        return self.sess.run(self.reconstruction, feed_dict={self.z_q:z_q})
        
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x:X})

    def save(self,sess,dir,step=None):
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,sess,model):
        self.saver.restore(sess,model)


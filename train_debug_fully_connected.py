#
# + cleaning version: TO BE trian.py
#
# + History
# 09/10 Regulate rotation angle only {0, 120, 240}
#
# import matplotlib
# matplotlib.use('Agg')

import os
import gc
import json
import time
import math
import argparse
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.profiler import ProfileOptionBuilder, Profiler
from tensorflow.python import debug as tf_debug

def get_args():
  p = argparse.ArgumentParser()
  p.add_argument(
    '--logdir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--figdir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--output_modeldir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--lr',
    type=float,
    default=0.001
  )
  p.add_argument(
    '--expname',
    type=str,
    default='new'
  )
  p.add_argument(
    '--num_epoch',
    type=int,
    default=5
  )
  p.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='number of pictures in minibatch'
  )
  p.add_argument(
    '--height',
    type=int,
    default=28
  )
  p.add_argument(
    '--width',
    type=int,
    default=28
  )
  p.add_argument(
    '--nblocks',
    type=int,
    default=3
  )
  p.add_argument(
    '--save_every',
    type=int,
    default=1
  )
  p.add_argument(
    '--copy_size',
    type=int,
    default=32
  )
  p.add_argument(
    '--dangle',
    type=int,
    default=1
  )
  p.add_argument(
    '--c_lambda',
    type=float,
    default=0.1
  )
  p.add_argument(
    '--rotation',
    action="store_true", 
    help='if user attaches this option, training images will be rondamly rotated',
    default=False
  )
  p.add_argument(
    '--debug',
    action='store_true',
  )
  args = p.parse_args()
  for f in args.__dict__:
    print("\t", f, (25 - len(f)) * " ", args.__dict__[f])
  print("\n")
  return args
  

def model_fn(shape=(28,28,1), nblocks=5, base_dim=3) :
    """
      Reference: https://blog.keras.io/building-autoencoders-in-keras.html
    """
    def convSeries_fn(x,
                      filters=16, 
                      kernel_size=3, 
                      nstack_layer=3, 
                      stride=2, 
                      up=True, 
                      pooling=True
                      ):
      """
      INPUT
        nstack_layer : number of iteration of conv layer before pooling
        up           : boolean. True is encoder, False is decoder(conv2D transpose)
      """
      for idx  in range(nstack_layer):
        if up:
          x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                     kernel_initializer='he_normal')(x)
        else:
          if idx == nstack_layer-1:
            x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, 
                                strides=(stride,stride), padding='same')(x)
          else:
            x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                     kernel_initializer='he_normal')(x)
        x = ReLU()(x)
          
      if pooling:
        x = MaxPooling2D((2, 2), padding='same')(x)
      # x = BatchNormalization()(x)
      return x

    # set params
    params = {
      'filters': [ 2**(i+base_dim) for i in range(nblocks)],
      'kernel_size': 3
    }
    
    ## start construction

    x = inp = Input(shape=shape, name='encoding_input')
    # x = Conv2D(filters=1, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    # encoder layers
    # for iblock in range(nblocks):
    #   filters = params["filters"][iblock]
    #   kernel_size = params["kernel_size"]
    #   if iblock != nblocks-1:
    #     x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, up=True)
    #   else:
    #     x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, up=True, pooling=False)
    # x = tf.reshape(x, shape=(-1,7,7,32))

    x = Flatten()(x)
    x = Dense(500)(x)
    x = ReLU()(x)
    # x = Dense(300)(x)
    # x = ReLU()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = tf.reshape(x, shape=(-1,2,2,32))

    # build model for encoder + digit layer
    encoder = Model(inp, x, name='encoder')
             
    x = inp = Input(x.shape[1:], name="decoder_input")
    x = Flatten()(x)
    # x = Dense(300)(x)
    # x = ReLU()(x)
    x = Dense(300)(x)
    x = ReLU()(x)
    # x = Dense(500)(x)
    # x = ReLU()(x)
    x = Dense(784)(x)
    x = ReLU()(x)
    x = tf.reshape(x, shape=(-1,28,28,1))
    # # decoder layers
    # for iblock in range(nblocks):
    #   filters = params["filters"][::-1][iblock]
    #   kernel_size = params["kernel_size"]
    #   if not iblock == nblocks-1:
    #     x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, up=False, pooling=False)
    #   else:
    #     x = convSeries_fn(x,filters=filters, kernel_size=kernel_size, up=True, pooling=False)
    #
    # x = Conv2D(filters=1, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    decoder = Model(inp, x, name='decoder')
             
    return encoder, decoder

def resize_image_fn(imgs,height=28, width=28):
  """Resize images from (#batch,H',W',C) --> (#batch,H,W,C)
  """
  reimgs = tf.image.resize_images(imgs, (height, width))
  return reimgs

def create_rotators(angles, image_shape=(28,28)):
    rotator_list = []
    for angle in angles:
      rotator_list.append(tf.contrib.image.angles_to_projective_transforms(
        angle, image_shape[0], image_shape[1] ) )
    return rotator_list


def rotate_with_rotators(imgs, rotators):
    output_imgs = tf.expand_dims(imgs,axis=0)
    for rotator in rotators:
        rotated_images = tf.contrib.image.transform(imgs, rotator)
        rotated_images = tf.expand_dims(rotated_images, axis=0)
        output_imgs = tf.concat((output_imgs, rotated_images), axis=0)
    return output_imgs

def encode_all_rotated_images(imgs, encoder):
    return tf.map_fn(encoder,imgs)

def reconstruct_all_rotated_images(imgs, encoder,decoder):
    codes = tf.map_fn(encoder,imgs)
    return tf.map_fn(decoder, codes)

def _random_choice(inputs, n_samples, random_choice_axis=0):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # stolen from stack overflow
    # (1, n_states) since multinomial requires 2D logits.
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

    ind = tf.multinomial(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, axis=random_choice_axis, name="random_choice")

def select_random_rotations(imgs, n_samples=32):
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(imgs)[0]), 0)
    ind = tf.multinomial(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)
    linear_indices = tf.constant(list(range(n_samples)), dtype=tf.int64)

    return tf.gather_nd(imgs,tf.transpose([ind,linear_indices]))

def input_fn(data, batch_size=32, copy_size=4, prefetch=1):
    """
      INPUT:
        prefetch: tf.int64. How many "minibatch" we asynchronously prepare on CPU ahead of GPU
    """
    # check batch/copy ratio
    try:
      if batch_size % copy_size == 0:
        print("\n Number of actual original images == {} ".format(int(batch_size)))
    except:
      raise ValueError("\n Division of batch size and copy size is not Integer \n")

    data1 = data.reshape(-1,28,28,1)
    dataset = tf.data.Dataset.from_tensor_slices((data1))
    dataset = dataset.shuffle(1000).repeat().batch(int(batch_size)).prefetch(prefetch)
    return dataset

def make_copy_rotate_image(oimgs_tf, batch_size=32, copy_size=4, height=28, width=28):
  """
    INPUT:
      oimgs_tf : original images in minibatch for rotation
    OUTPUT:
      crimgs: minibatch with original and these copy + rotations
  """
  print(" SHAPE in make_copy_rotate_image", oimgs_tf.shape)
  # operate within cpu   
  stime = datetime.now()
  img_list = []
  for idx in range(int(batch_size)):
    tmp_img_tf = oimgs_tf[idx]
    img_list.extend([tf.reshape(tmp_img_tf, (1,height,width,1))] )
    # img_list.extend([ tf.expand_dims(tf.identity(tmp_img_tf), axis=0) for i in range(copy_size-1)])

  images = tf.concat(img_list, axis=0)
  # images = coimgs
  # crimgs = rotate_fn(coimgs, seed=np.random.randint(0,999), return_np=False)
  etime = datetime.now()
  print(" make_copy_rotate {} s".format(etime - stime))
  return images

def reconstruction_loss(inputs, outputs):
    outputs = tf.expand_dims(outputs, axis=0)
    all_rotation_l2_loss = tf.reduce_sum(tf.square(outputs-inputs),axis=(2,3,4))
    min_across_angles = tf.math.reduce_min(all_rotation_l2_loss,axis=0)
    return min_across_angles

def latent_space_loss(z_hat, all_latent_codes):
    z_hat = tf.expand_dims(z_hat, axis=0)
    all_rotation_latent_loss = tf.reduce_sum(tf.square(z_hat-all_latent_codes),axis=(2,3,4))
    max_across_angles = tf.math.reduce_max(all_rotation_latent_loss,axis=0)
    return max_across_angles

if __name__ == '__main__':
  # time for data preparation
  prep_stime = time.time()

  # get arg-parse as FLAGS
  FLAGS = get_args()

  # set data
  mnist = input_data.read_data_sets(os.path.abspath("./MNIST_data/"), one_hot=False)
  train_images  = mnist.train.images

  # ad-hoc params
  # num_test_images = int(FLAGS.batch_size/FLAGS.copy_size)
  num_test_images = FLAGS.batch_size
  
  # make dirs
  os.makedirs(FLAGS.logdir, exist_ok=True)
  os.makedirs(FLAGS.figdir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir, exist_ok=True)
  os.makedirs(FLAGS.output_modeldir+'/timelines', exist_ok=True)

  # outputnames
  ctime = datetime.now()
  bname1 = '_nepoch-'+str(FLAGS.num_epoch)+'_lr-'+str(FLAGS.lr)
  bname2 = '_nbatch-'+str(FLAGS.batch_size)+'_lambda'+str(FLAGS.c_lambda)+'_dangle'+str(FLAGS.dangle)
  figname   = 'fig_'+FLAGS.expname+bname1+bname2+str(ctime.strftime("%s"))
  ofilename = 'loss_'+FLAGS.expname+bname1+bname2+str(ctime.strftime("%s"))+'.txt'
  dfilename = 'degree_'+FLAGS.expname+bname1+bname2+str(ctime.strftime("%s"))+'.txt'

  # set global time step
  global_step = tf.train.get_or_create_global_step()

  with tf.device('/CPU'):
    # get dataset and one-shot-iterator
    dataset = input_fn(train_images, 
                     batch_size=FLAGS.batch_size, 
                     copy_size=FLAGS.copy_size
    )
    # apply preprocessing  
    dataset_mapper = dataset.map(lambda x: make_copy_rotate_image(
            x,batch_size=FLAGS.batch_size,copy_size=FLAGS.copy_size,
            height=FLAGS.height,width=FLAGS.width
        )
    )

  # why get_one_shot_iterator leads OOM error?
  train_iterator = dataset_mapper.make_initializable_iterator()
  imgs  = train_iterator.get_next()
  print(imgs)

  # exit()
  # get model
  encoder, decoder = model_fn(nblocks=FLAGS.nblocks)
  print("\n {} \n".format(encoder.summary()), flush=True)
  print("\n {} \n".format(decoder.summary()), flush=True)
  # exit()
  test_angles = [0,120,240]
  image_shape = (28,28)
  experimental_rotators = create_rotators(angles=test_angles, image_shape=image_shape)
  all_rotations = rotate_with_rotators(imgs, experimental_rotators)
  all_latent_codes = encode_all_rotated_images(all_rotations, encoder)
  selected_rotations = select_random_rotations(all_rotations)
  z_hat = encoder(selected_rotations)
  x_hat = decoder(z_hat)
  recon_loss = reconstruction_loss(all_rotations, x_hat)
  latent_loss = latent_space_loss(z_hat, all_latent_codes)
  # recon_loss = reconstruction_loss(all_rotations, selected_rotations)
  # latent_loss = latent_space_loss(selected_rotations, all_rotations)
  total_loss = tf.reduce_mean(recon_loss + FLAGS.c_lambda * latent_loss)
  # print(x_hat)
  # print(selected_rotations)
  # print(total_loss)

  with tf.Session() as sess:
    #TODO: Erase Comment off if debugging is necessay
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # initial run
    init=tf.global_variables_initializer()
    sess.run(init)
    sess.run(train_iterator.initializer)
    testytest = sess.run(latent_loss)
    print(testytest)
    recon_test = sess.run(recon_loss)
    print(recon_test)
    # inputs = sess.run(imgs)
    # print(np.max(inputs[:]))
    # print(np.min(inputs[:]))
    # outputs = sess.run(x_hat)
    # print(np.max(inputs[:]))
    # print(np.min(inputs[:]))

  # exit()
  # loss + optimizer
  # compute loss and train_ops

 
  # Apply optimization 
  """
    # Full version
    #   Compute mean-loss_1st term + lambda * mean-loss 2nd term
  """
  train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(total_loss)
  #      tf.reduce_mean(loss_rotate)

  # L2 loss: Check Minibatch creation
  #train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(loss)
 

  # Reconst-agn version
  # 09/17 method-1 take mean
  #train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(
  #      tf.reduce_mean(loss_reconst),
  #)

  # Bottleneck version
  #train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(
  #      tf.multiply( tf.constant(FLAGS.c_lambda, dtype=tf.float32), tf.reduce_mean(loss_rotate))
  #)


  # observe loss values with tensorboard
  with tf.name_scope("summary"):
    #tf.summary.scalar("L2 loss", loss )
    tf.summary.scalar("reconst loss", tf.reduce_mean(recon_loss) )
    tf.summary.scalar("rotate loss",  
        tf.multiply(tf.constant(FLAGS.c_lambda,dtype=tf.float32), tf.reduce_mean(latent_loss))
    )
    merged = tf.summary.merge_all()

  # set-up save models
  save_models = {"encoder": encoder, "decoder": decoder}

  # save model definition
  # for m in save_models:
  #   with open(os.path.join(FLAGS.output_modeldir, m+'.json'), 'w') as f:
  #     f.write(save_models[m].to_json())

  # gpu config 
  config = tf.ConfigProto(
    # gpu_options=tf.GPUOptions(
    #     allow_growth=True
    # ),
    log_device_placement=False
  )

  # End for prep-processing during main code
  prep_etime = (time.time() -prep_stime)/60.0 # minutes
  print("\n### Entering Training Loop ###\n")
  print("   Data Pre-Processing time [minutes]  : %f" % prep_etime, flush=True)
  

  # TRAINING
  with tf.Session(config=config) as sess:
    #TODO: Erase Comment off if debugging is necessay
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # initial run
    init=tf.global_variables_initializer()
    sess.run(init)
    sess.run(train_iterator.initializer)

    # initialize other variables
    num_batches=int(len(train_images)*FLAGS.copy_size)//FLAGS.batch_size
    angle_list = [0,120,240]
    loss_l2_list = []
    loss_reconst_list = []
    loss_rotate_list = []
    deg_reconst_list = []

    # Trace and Profiling options
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_modeldir, 'logs'), sess.graph) 
    run_metadata = tf.RunMetadata()
    run_opts = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)

    #====================================================================
    # Training
    #====================================================================
    stime = time.time()
    for epoch in range(FLAGS.num_epoch):
      for iteration in range(num_batches):
        _, tf.summary = sess.run([train_ops, merged])

        if iteration % 100 == 0:
          _loss_reconst,_loss_rotate, _theta_reconst = sess.run(
              [recon_loss, latent_loss, x_hat]
          )
          print(
                 "iteration {:7} | loss reconst {:10}  loss rotate {:10}".format(
              iteration,
              np.mean(_loss_reconst),
              np.mean(_loss_rotate)
            ), flush=True
          )
          # Save loss to lsit
          loss_reconst_list.append(np.mean(_loss_reconst))
          loss_rotate_list.append(np.mean(_loss_rotate))
          deg_reconst_list.append(_loss_reconst)

      # save model at every N steps
      if epoch % FLAGS.save_every == 0:
         for m in save_models:
           save_models[m].save_weights(
             os.path.join(
               FLAGS.output_modeldir, "{}-{}.h5".format(m, epoch)
             )
           )


         # Full
         _loss_reconst,_loss_rotate, _theta_reconst = sess.run(
              [recon_loss, latent_loss, x_hat]
         )
         print( "\n Save Model Epoch {}: \n 1st term Loss: {} 2nd term Loss: {} \n".format(
              epoch, np.mean(_loss_reconst), np.mean(_loss_rotate)
            ),
            flush=True
         )

         # L2
         #_loss_l2 = sess.run(loss)
         #     #angle_list[np.argmax(_loss_rotate)],
         #print( "\n Save Model Epoch {}: Loss {:12} \n".format(
         #     epoch,
         #     _loss_l2,
         #   )
         #)
         

    #=======================
    # + Visualization
    #=======================
      with tf.device("/CPU"):
        results, test_images, rtest_images = sess.run(
          [x_hat, imgs, selected_rotations]
        )

        #Comparing original images with reconstructions
        f,a=plt.subplots(3,num_test_images,figsize=(2*num_test_images,6))
        for idx, i in enumerate(range(num_test_images)):
          a[0][idx].imshow(np.reshape(test_images[i],(FLAGS.height,FLAGS.width)), cmap='jet')
          a[1][idx].imshow(np.reshape(rtest_images[i],(FLAGS.height,FLAGS.width)), cmap='jet')
          a[2][idx].imshow(np.reshape(results[i],(FLAGS.height,FLAGS.width)), cmap='jet')
          # set axis turn off
          a[0][idx].set_xticklabels([])
          a[0][idx].set_yticklabels([])
          a[1][idx].set_xticklabels([])
          a[1][idx].set_yticklabels([])
          a[2][idx].set_xticklabels([])
          a[2][idx].set_yticklabels([])
        plt.savefig(FLAGS.figdir+'/'+figname+'.png')

      ## loss L2
      #with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
      #  for r in zip(loss_l2_list):
      #    f.write(str(r)+'\n')

      ## Full
      # loss
        with open(os.path.join(FLAGS.logdir, ofilename), 'w') as f:
          for re, ro in zip(loss_reconst_list, loss_rotate_list):
            f.write(str(re)+','+str(ro)+'\n')
        # degree
        with open(os.path.join(FLAGS.logdir, dfilename), 'w') as f:
          f.write("\n".join(" ".join(map(str,x)) for x in (deg_reconst_list, deg_reconst_list)))


  print("### DEBUG NORMAL END ###")
  # FINISH
  etime = (time.time() -stime)/60.0 # minutes
  print("   Execution time [minutes]  : %f" % etime, flush=True)

#
# + cleaning version: TO BE trian.py
#
# + History
# 09/10 Regulate rotation angle only {0, 120, 240}
#
import matplotlib
matplotlib.use('Agg')

import os
import gc
import glob
import json
import time
import math
import argparse
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from clouds_models import modified_convolutional_architecture as model_fn
from datetime import datetime
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.profiler import ProfileOptionBuilder, Profiler
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.data import parallel_interleave
from tensorflow.contrib.data import batch_and_drop_remainder

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
    '--input_datadir',
    type=str,
    default='./clouds_tfdata/'
  )
  p.add_argument(
    '--output_modeldir',
    type=str,
    default='./'
  )
  p.add_argument(
    '--lr',
    type=float,
    default=0.01
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
    default=8,
    help='number of pictures in minibatch'
  )
  p.add_argument(
    '--height',
    type=int,
    default=128
  )
  p.add_argument(
    '--width',
    type=int,
    default=128
  )
  p.add_argument(
    '--nblocks',
    type=int,
    default=3
  )
  p.add_argument(
      '--global_normalization',
      action="store_true",
      help='normalize cloud data if data was not normalized to mean 0 stdv 1',
      default=True
  )
  p.add_argument(
      '--stats_datadir',
      type=str,
      default='./clouds_tfdata/global_mean_std/'
  )
  p.add_argument(
    '--save_every',
    type=int,
    default=1
  )
  p.add_argument(
    '--copy_size',
    type=int,
    default=8
  )
  p.add_argument(
    '--dangle',
    type=int,
    default=1
  )
  p.add_argument(
    '--c_lambda',
    type=float,
    default=0.05
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

def select_random_rotations(imgs, n_samples=8):
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(imgs)[0]), 0)
    ind = tf.multinomial(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)
    linear_indices = tf.constant(list(range(n_samples)), dtype=tf.int64)

    return tf.gather_nd(imgs,tf.transpose([ind,linear_indices]))

def input_fn(data, batch_size=8, copy_size=4, prefetch=1):
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

    data1 = data.reshape(-1,128,128,1)
    dataset = tf.data.Dataset.from_tensor_slices((data1))
    dataset = dataset.shuffle(1000).repeat().batch(int(batch_size)).prefetch(prefetch)
    return dataset


def input_clouds_fn(filelist, gmean, gstdv, batch_size=32, copy_size=4, prefetch=1, read_threads=4, distribute=(1, 0)):
    """
      INPUT:
        prefetch: tf.int64. How many "minibatch" we asynchronously prepare on CPU ahead of GPU
    """

    def parser(ser):
        """
        Decode & Pass datast in tf.record
        *Cuation*
        floating point: tfrecord data ==> tf.float64
        """
        features = {
            "shape": tf.FixedLenFeature([3], tf.int64),
            "patch": tf.FixedLenFeature([], tf.string),
            "filename": tf.FixedLenFeature([], tf.string),
            "coordinate": tf.FixedLenFeature([2], tf.int64),
        }
        decoded = tf.parse_single_example(ser, features)
        patch = tf.reshape(
            tf.decode_raw(decoded["patch"], tf.float64), decoded["shape"]
            # tf.decode_raw(decoded["patch"], tf.float32), decoded["shape"]
        )
        print("shape check in pipeline {}".format(patch.shape), flush=True)
        # patch = tf.random_crop(patch, shape)
        # return decoded["filename"], decoded["coordinate"], patch

        # conversion of tensor
        patch = tf.cast(patch, tf.float32)
        if not gstdv.all() == 0.00:
            # np to tf
            gmean_tf = tf.constant(gmean, dtype=tf.float32)
            gstdv_tf = tf.constant(gstdv, dtype=tf.float32)
            # avoid 0 div
            patch -= gmean_tf
            patch /= gstdv_tf
            print("\n## Normalization process Done ##\n")

        return patch

    # check batch/copy ratio
    try:
        if batch_size % copy_size == 0:
            print("\n Number of actual original images == {} ".format(int(batch_size)))
    except:
        raise ValueError("\n Division of batch size and copy size is not Integer \n")

    dataset = (
        tf.data.Dataset.list_files(filelist, shuffle=True)
            .shard(*distribute)
            .apply(
            parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f).map(parser),
                cycle_length=read_threads,
                sloppy=True,
            )
        )
    )
    dataset = dataset.shuffle(1000).repeat().batch(int(batch_size)).prefetch(prefetch)
    return dataset


def make_copy_rotate_image(oimgs_tf, batch_size=32, copy_size=4, height=128, width=128):
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
    #img_list.extend([tf.reshape(tmp_img_tf, (1,height,width,1))] )
    img_list.extend([tf.reshape(tmp_img_tf, (1,height,width,6))] )
    # img_list.extend([ tf.expand_dims(tf.identity(tmp_img_tf), axis=0) for i in range(copy_size-1)])

  images = tf.concat(img_list, axis=0)
  # images = coimgs
  # crimgs = rotate_fn(coimgs, seed=np.random.randint(0,999), return_np=False)
  etime = datetime.now()
  print(" make_copy_rotate {} s".format(etime - stime))
  return images

def reconstruction_loss(inputs, outputs):
    outputs = tf.expand_dims(outputs, axis=0)
    all_rotation_l2_loss = tf.reduce_mean(tf.square(outputs-inputs),axis=(2,3,4))
    min_across_angles = tf.math.reduce_min(all_rotation_l2_loss,axis=0)
    return min_across_angles

def latent_space_loss(z_hat, all_latent_codes):
    z_hat = tf.expand_dims(z_hat, axis=0)
    all_rotation_latent_loss = tf.reduce_mean(tf.square(z_hat-all_latent_codes),axis=(2,3,4))
    max_across_angles = tf.math.reduce_max(all_rotation_latent_loss,axis=0)
    return max_across_angles

if __name__ == '__main__':
  # time for data preparation
  prep_stime = time.time()

  # get arg-parse as FLAGS
  FLAGS = get_args()

  # get filenames of training data as list
  train_images_list = glob.glob(os.path.abspath(FLAGS.input_datadir)+'/*.tfrecord')

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

  # print()
  if FLAGS.global_normalization:
      global_mean = np.load(glob.glob(FLAGS.stats_datadir + '/*_gmean.npy')[0])
      global_stdv = np.load(glob.glob(FLAGS.stats_datadir + '/*_gstdv.npy')[0])
  else:
      global_mean = np.zeros((6))
      global_stdv = np.ones((6))

  # set global time step
  global_step = tf.train.get_or_create_global_step()

  with tf.device('/CPU'):
    # get dataset and one-shot-iterator
    dataset = input_clouds_fn(train_images_list,
                              global_mean,
                              global_stdv,
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
  encoder, decoder = model_fn()
  # print("\n {} \n".format(encoder.summary()), flush=True)
  # print("\n {} \n".format(decoder.summary()), flush=True)
  # exit()
  test_angles = [0,120,240]
  #image_shape = (28,28)
  image_shape = (128,128)
  experimental_rotators = create_rotators(angles=test_angles, image_shape=image_shape)
  all_rotations = rotate_with_rotators(imgs, experimental_rotators)
  all_latent_codes = encode_all_rotated_images(all_rotations, encoder)
  selected_rotations = select_random_rotations(all_rotations)
  z_hat = encoder(selected_rotations)
  x_hat = decoder(z_hat)
  recon_loss = reconstruction_loss(all_rotations, x_hat)
  latent_loss = latent_space_loss(z_hat, all_latent_codes)
  total_loss = tf.reduce_mean(recon_loss + FLAGS.c_lambda * latent_loss)

  # check loss for debugging? 
  # with tf.Session() as sess:
  #   # initial run
  #   init=tf.global_variables_initializer()
  #   sess.run(init)
  #   sess.run(train_iterator.initializer)
  #   testytest = sess.run(imgs)
  #   print(testytest)
  #   print(np.max(testytest[:]))
  #   print(np.min(testytest[:]))
  #   print(np.shape(testytest))
  #   # exit()
  #   recon_test = sess.run(recon_loss)
  #   print(recon_test)
 
  # Apply optimization 
  """
    # Full version
    #   Compute mean-loss_1st term + lambda * mean-loss 2nd term
  """
  train_ops = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(total_loss)
  #      tf.reduce_mean(loss_rotate)


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
    # intra_op_parallelism_threads=NUM_THREADS,
    # inter_op_parallelism_threads=NUM_THREADS,
    # allow_soft_placement=True,
    log_device_placement=False
  )

  # End for prep-processing during main code
  prep_etime = (time.time() -prep_stime)/60.0 # minutes
  print("\n### Entering Training Loop ###\n")
  print("   Data Pre-Processing time [minutes]  : %f" % prep_etime, flush=True)
  

  # TRAINING
  with tf.Session(config=config) as sess:
    # initial run
    init=tf.global_variables_initializer()
    sess.run(init)
    sess.run(train_iterator.initializer)

    # initialize other variables
    """
    Num of batches ==> one tf.record has 10,000 cloud patches
    """
    num_batches=int(len(train_images_list)*FLAGS.copy_size*10000)//FLAGS.batch_size
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

        if iteration % 10 == 0:
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
          #exit()

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

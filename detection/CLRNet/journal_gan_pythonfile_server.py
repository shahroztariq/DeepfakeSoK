
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import backend as K

import os
# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     import h5py
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, Input
from keras.layers import Dense
from keras.models import Model
sys.stderr = stderr
from keras.utils.multi_gpu_utils import multi_gpu_model
import tensorflow as tf
from sklearn.metrics import classification_report
from keras.applications import NASNetLarge
from utility_classes import DataGenerator, make_partition, get_available_gpus, shallow_cnn, cpd64, nincnn, autoencoder, meso4, Imgaug_DataGenerator
from keras.applications import Xception, inception_resnet_v2, VGG19, NASNetLarge
from keras_applications import resnext
import multiprocessing
from time import gmtime, strftime
import imgaug as ia
import imgaug.augmenters as iaa
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
import platform
import numpy as np
import glob
from sklearn.preprocessing import OneHotEncoder

print("Starting GPU configurations...")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# Choose GPU NUMBERS [0, 1, 2, 3]
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
sess.run(tf.global_variables_initializer())
print("GPU configurations done!")

# tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# current_file_name = os.path.basename(__file__)[:-3]

# Parameters
img_size = 128
params = {'model_sel': 'xception', # xception / resnext50 / inception_resnet_v2 / vgg19 / nincnn / autoencoder / mesonet / NASNetLarge
          'model_imagenet': True,
          'model_mesonet': False,
          'augmentation': 'imgaug',  # no / keras / imgaug / autoaugment
          'dim': (img_size, img_size),
          'n_channels': 3,
          'batch_size': 32,
          'gpu': 1,
          'device_name': platform.node(),
          'cpu': multiprocessing.cpu_count(),
          'epochs': 200,
          'checkpoint_dir': os.path.join('/home/sangyup/projects/journal_gan/training/checkpoints_journal',
                                         strftime("%Y-%m-%d-%H-%M-%S", gmtime())),
          # 'file_name': current_file_name,
          'n_classes': 2,
          'shuffle': True,
          # 'train_directory': "/home/sangyup/projects/journal_gan/dataset/no_SR/train",
          'train_directory': "/home/sangyup/projects/journal_gan/Progressive-Face-Super-Resolution_KAIST/dataset/train",
          # 'train_directory': "/home/sangyup/projects/journal_gan/dataset/dataset_512/train",
          # "datasets/{}x{}/train".format(img_size, img_size),
          # 'test_directory': "/home/sangyup/projects/journal_gan/dataset/no_SR/test",
          'test_directory': "/home/sangyup/projects/journal_gan/Progressive-Face-Super-Resolution_KAIST/dataset/test",
          # 'test_directory': "/home/sangyup/projects/journal_gan/dataset/dataset_512/test",
          # "datasets/{}x{}/test".format(img_size, img_size),
          # 'validation_directory': "/home/sangyup/projects/journal_gan/dataset/no_SR/validation",
          'validation_directory': "/home/sangyup/projects/journal_gan/Progressive-Face-Super-Resolution_KAIST/dataset/validation",
          # 'validation_directory': "/home/sangyup/projects/journal_gan/dataset/dataset_512/validation",
          # 'validation_directory': "/home/sangyup/projects/journal_gan/dataset/celeba_val",
          'class_mode': 'categorical',  # if it's autoencoder --> input / otherwise, categorical, binary
          'image_type': 'jpg',
          'period_checkpoint': 1,
          'optimizer': optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
          'loss': 'binary_crossentropy',
          # 'classes': ['real']
          'classes': ['fake', 'real']
}
if not os.path.exists(params['checkpoint_dir']):
    os.makedirs(strftime(params['checkpoint_dir']))
print(params)

# Generators

print("Loading Training data... Augmentation : ", params['augmentation'])
if params['augmentation'] == 'no':
    train_datagen = ImageDataGenerator(  # No augmentation
        rescale=1. / 255,
    )
elif params['augmentation'] == 'keras':
    train_datagen = ImageDataGenerator(  # Keras augmentation
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        #     fill_mode='nearest',
    )

elif params['augmentation'] == 'imgaug':
    labels = list()
    image_paths = list()
    fake_image_paths = glob.glob(params['train_directory'] + '/fake/*.jpg')
    labels.extend(np.zeros(len(fake_image_paths)))
    real_image_paths = glob.glob(params['train_directory'] + '/real/*.jpg')
    labels.extend(np.ones(len(real_image_paths)))
    labels = np.array(labels).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels = enc.transform(labels).toarray()
    image_paths.extend(fake_image_paths)
    image_paths.extend(real_image_paths)
    print(len(image_paths))
    # print(labels)
    training_generator = Imgaug_DataGenerator(image_paths, labels, image_dimensions=(img_size, img_size, 3), batch_size=params['batch_size'], augment=True, shuffle=True)


elif params['augmentation'] == 'autoaugment':
    print("Not yet done")

val_datagen = ImageDataGenerator(rescale=1. / 255)

if not params['augmentation'] == 'imgaug':
    training_generator = train_datagen.flow_from_directory(
        params['train_directory'],  # this is the target directory
        target_size=params['dim'],  # all images will be resized to dim
        batch_size=params['batch_size'],
        class_mode=params['class_mode'],
        classes=params['classes'],
        shuffle=True,
        seed=1,
    )

print("Loading Validation data...")
validation_generator = val_datagen.flow_from_directory(
    params['validation_directory'],
    target_size=params['dim'],
    batch_size=params['batch_size'],
    class_mode=params['class_mode'],
    classes=params['classes'],
    shuffle=True,
    seed=1
)

# Loading the model

# model_input = Input(shape=(*params['dim'], params['n_channels']))
inputs = Input(shape=(), name='image_input')
if params['gpu'] <= 1:
    print("[INFO] training with {} CPUs & {} GPU...".format(params['cpu'], params['gpu']))

    if params['model_sel'] == 'nincnn':
        model_name = "nincnn"
        model = nincnn(model_input=inputs)
    elif params['model_sel'] == 'autoencoder':
        model_name = "autoencoder"
        model = autoencoder(model_input=inputs)
        print(model.summary())
    elif params['model_sel'] == 'mesonet':
        model_name = "mesonet"
        model = meso4(model_input=inputs)
        if params['model_mesonet']:
            print("Loading Meso4 pretrained")
            model.load_weights('Meso4_DF')
    elif params['model_sel'] == 'NASNetLarge':
        if not params['model_imagenet']:
            model_name = "NASNetLarge"
            model = Xception(include_top=True,
                             weights=None,  # Change to use imagenet
                             input_shape=(*params['dim'], params['n_channels']),
                             classes=params['n_classes'],
                             pooling='avg')
    elif params['model_sel'] == 'xception':
        if not params['model_imagenet']:
            model_name = "xception"
            model = Xception(include_top=True,
                             weights=None,  # Change to use imagenet
                             input_shape=(*params['dim'], params['n_channels']),
                             classes=params['n_classes'],
                             pooling='avg')
            # print(model.summary())
        else:
            # LOAD WITH WEIGHTS
            model_name = "xception"

            model = Xception(include_top=True,
                             weights=None,  # Change to use imagenet
                             input_tensor=inputs,
                             # classes=params['n_classes'],
                             pooling='avg')
            # model.summary()
            weights_file = "/home/sangyup/projects/journal_gan/imagenet_pretrained_models/xception_weights_tf_dim_ordering_tf_kernels.h5"
            model.load_weights(weights_file)
            model.layers.pop()
            x = model.layers[-1].output
            x = Dense(params['n_classes'], activation='linear', name='predictions', kernel_initializer="he_normal")(x)
            model = Model(inputs=inputs, outputs=x)
            # model.summary()
    elif params['model_sel'] == 'resnext50':
        if not params['model_imagenet']:
            model_name = "resnext50"
            model = resnext.ResNeXt50(include_top=True,
                                          weights=None,  # Change to use imagenet
                                          input_shape=(*params['dim'], params['n_channels']),
                                          classes=params['n_classes'],
                                          pooling='avg',
                                          backend = keras.backend,
                                          layers = keras.layers,
                                          models = keras.models,
                                          utils = keras.utils)
    elif params['model_sel'] == 'inception_resnet_v2':
        if not params['model_imagenet']:
            model_name = "inception_resnet_v2"
            model = inception_resnet_v2.InceptionResNetV2(include_top=True,
                                                          weights=None,  # Change to use imagenet
                                                          input_shape=(*params['dim'], params['n_channels']),
                                                          classes=params['n_classes'],
                                                          pooling='avg')
        else:
            model = inception_resnet_v2.InceptionResNetV2(include_top=True,
                                                          weights=None,  # Change to use imagenet
                                                          input_tensor=inputs,
                                                          classes=params['n_classes'],
                                                          pooling='avg')
            model.summary()
            weights_file = "D:\labwork\local_ganwork\journal_gan\imagenet_pretrained_models\inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
            model.load_weights(weights_file)
            model.layers.pop()
            x = model.layers[-1].output
            x = Dense(params['n_classes'], activation='linear', name='predictions', kernel_initializer="he_normal")(x)
            model = Model(inputs=inputs, outputs=x)
            model.summary()
    elif params['model_sel'] == 'vgg19':
        if not params['model_imagenet']:
            model_name = "vgg19"
            model = VGG19(include_top=True,
                          weights=None,  # Change to use imagenet
                          input_shape=(*params['dim'], params['n_channels']),
                          classes=params['n_classes'],
                          pooling='avg')

else:
    print("[INFO] training with {} CPUs & {} GPUs...".format(params['cpu'], params['gpu']))
    with tf.device("/cpu:0"):
        print("[INFO] training with {} CPUs & {} GPU...".format(params['cpu'], params['gpu']))

        if params['model_sel'] == 'nincnn':
            model_name = "nincnn"
            model = nincnn(model_input=inputs)
        elif params['model_sel'] == 'autoencoder':
            model_name = "autoencoder"
            model = autoencoder(model_input=inputs)
        elif params['model_sel'] == 'mesonet':
            model_name = "mesonet"
            model = meso4(model_input=inputs)
            if params['model_mesonet']:
                print("Loading Meso4 pretrained")
                model.load_weights('Meso4_DF')
        elif params['model_sel'] == 'NASNetLarge':
            if not params['model_imagenet']:
                model_name = "NASNetLarge"
                model = Xception(include_top=True,
                                 weights=None,  # Change to use imagenet
                                 input_shape=(*params['dim'], params['n_channels']),
                                 classes=params['n_classes'],
                                 pooling='avg')
        elif params['model_sel'] == 'xception':
            if not params['model_imagenet']:
                model_name = "xception"
                model = Xception(include_top=True,
                                 weights=None,  # Change to use imagenet
                                 input_shape=(*params['dim'], params['n_channels']),
                                 classes=params['n_classes'],
                                 pooling='avg')
            else:
                # LOAD WITH WEIGHTS
                model_name = "xception"

                model = Xception(include_top=True,
                                 weights=None,  # Change to use imagenet
                                 input_tensor=inputs,
                                 # classes=params['n_classes'],
                                 pooling='avg')
                model.summary()
                weights_file = "D:\labwork\local_ganwork\journal_gan\imagenet_pretrained_models/xception_weights_tf_dim_ordering_tf_kernels.h5"
                model.load_weights(weights_file)
                model.layers.pop()
                x = model.layers[-1].output
                x = Dense(params['n_classes'], activation='linear', name='predictions', kernel_initializer="he_normal")(
                    x)
                model = Model(inputs=inputs, outputs=x)
                model.summary()
        elif params['model_sel'] == 'resnext50':
            if not params['model_imagenet']:
                model_name = "resnext50"
                model = resnext.ResNeXt50(include_top=True,
                                          weights=None,  # Change to use imagenet
                                          input_shape=(*params['dim'], params['n_channels']),
                                          classes=params['n_classes'],
                                          pooling='avg',
                                          backend=keras.backend,
                                          layers=keras.layers,
                                          models=keras.models,
                                          utils=keras.utils)
        elif params['model_sel'] == 'inception_resnet_v2':
            if not params['model_imagenet']:
                model_name = "inception_resnet_v2"
                model = inception_resnet_v2.InceptionResNetV2(include_top=True,
                                                              weights=None,  # Change to use imagenet
                                                              input_shape=(*params['dim'], params['n_channels']),
                                                              classes=params['n_classes'],
                                                              pooling='avg')
            else:
                model = inception_resnet_v2.InceptionResNetV2(include_top=True,
                                                              weights=None,  # Change to use imagenet
                                                              input_tensor=inputs,
                                                              classes=params['n_classes'],
                                                              pooling='avg')
                model.summary()
                weights_file = "D:\labwork\local_ganwork\journal_gan\imagenet_pretrained_models\inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
                model.load_weights(weights_file)
                model.layers.pop()
                x = model.layers[-1].output
                x = Dense(params['n_classes'], activation='linear', name='predictions', kernel_initializer="he_normal")(
                    x)
                model = Model(inputs=inputs, outputs=x)
                model.summary()
        elif params['model_sel'] == 'vgg19':
            if not params['model_imagenet']:
                model_name = "vgg19"
                model = VGG19(include_top=True,
                              weights=None,  # Change to use imagenet
                              input_shape=(*params['dim'], params['n_channels']),
                              classes=params['n_classes'],
                              pooling='avg')

    model = multi_gpu_model(model=model, gpus=params['gpu'])
# print(model.summary())

# Checkpoint configuration

checkpoints = ModelCheckpoint(
    os.path.join(params['checkpoint_dir'], model_name + '_{epoch:03d}' + '.hd5f'),
    save_weights_only=True,
    period=params['period_checkpoint'])
csv_logger = CSVLogger(os.path.join(params['checkpoint_dir'], 'log.csv'), append=True, separator=';')
file = open(os.path.join(params['checkpoint_dir'], 'parameters'), "w")
file.write(str(params))
file.close()

# Train the model

model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])

# Train model on dataset

if params['augmentation'] == 'imgaug':
    length_steps = len(training_generator)
else:
    length_steps = len(training_generator.filenames) // params['batch_size']
print("length_steps:", length_steps)
# print(training_generator.__getitem__(0))

g = model.fit_generator(generator=training_generator,
                        steps_per_epoch=length_steps,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator.filenames) // params['batch_size'],
                        epochs=params['epochs'],
                        callbacks=[checkpoints, csv_logger],
                        shuffle=params['shuffle'],
                        verbose=2)

# use_multiprocessing=True, workers=params['cpu'])
file = open(os.path.join(params['checkpoint_dir'], 'history'), "w")
file.write(str(g.history))
file.close()
print(g.history)

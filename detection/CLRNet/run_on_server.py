import os
import cv2
import glob
import numpy as np
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import Sequence, to_categorical

from resnet_convlstm import ResNet
from tensorflow.python.keras.utils.layer_utils import print_summary
from datetime import datetime as dt
print(cv2.__version__)
from PIL import Image
import random
from sklearn.utils import class_weight
random.seed(32)
dataset_dir = 'DeepFakeDatasetReal'


#   Here, `x_set` is list of path to the images,
#   `y_set` are the associated classes,
#   'v' is the no. of videos for each batch
#   and fpv is the no. of frames from each video
class DFVDSequence(Sequence):

    def __init__(self, x_set, y_set, v, fpv):
        self.x, self.y = x_set, y_set
        self.batch_size = fpv * v
        self.count = 0
        self.video_iterator = 0
        self.frame_iterator = 0
        self.v = v
        self.fpv = fpv
        self.dataset_size = sum([len(value) for key, value in self.x.items()])
        self.frame_counter = 0
        self.on_epoch_end()

    def __len__(self):
        # size of dataset_size divided by batchsize
        return int(np.ceil(self.dataset_size / float(self.batch_size)))

    def __getitem__(self, idx):
        # batch_x = np.empty((self.v,self.fpv, 128,128, 3))
        batch_x = []
        batch_y = []
        video_idx = 0
        # print('enter')
        i = 0
        while i < self.v:
            i += 1
            if self.frame_counter >= self.dataset_size:
                print('break')

                break

            frames2read = self.x[str(self.video_iterator)][self.frame_iterator:self.frame_iterator + self.fpv]
            # print(len(frames2read))
            # frame_idx=0
            temp_frames = []
            if len(frames2read) >= self.fpv:
                for frame in frames2read:
                    # print('Shape:',np.array(Image.open(frame)).shape)
                    temp_frames.append(np.asarray(Image.open(frame))/255.0)
                    # batch_x[video_idx,frame_idx,:,:,:] = np.asarray(Image.open(frame))
                    # batch_x[video_idx] np.asarray(Image.open(frame))
                    # print("batch_x:",batch_x[video_idx,frame_idx,].shape)
                    # img = Image.fromarray(np.uint8(batch_x[video_idx,frame_idx,:,:]), 'RGB')
                    # img.save('test.jpg')
                    # frame_idx+=1
                    # print(self.y[video_idx],frame)
                # print(batch_x.shape)
                # print(frames2read)
                batch_x.append(temp_frames)
                # print(video_idx)
                batch_y.append(self.y[video_idx])
            # else:
            #     i-=1
            # self.dataset_size-=len(frames2read)
            #     print('do something')
            # print(np.array(batch_x).shape)
            self.video_iterator += 1
            self.frame_counter += len(frames2read)
            if self.video_iterator % len(self.x) == 0:
                self.video_iterator = 0
                self.frame_iterator += len(frames2read)
            video_idx += 1
        if self.video_iterator % len(self.x) == 0:
            self.video_iterator = 0
        # print(self.frame_counter)
        # print(self.dataset_size)
        # print("")
        # counter=0
        # for video in batch_x:
        #     for frame in video:
        #         img = Image.fromarray(np.uint8(frame), 'RGB')
        #         img.save(str(counter)+'_'+str(batch_y[counter])+'_my.png')
        #         counter+=1
        # img.show()
        # print(batch_x.shape)
        if len(batch_y) > 0:
            batch_y = to_categorical(batch_y)
            batch_x = np.array(batch_x)
        # print(batch_x.shape,batch_y.shape)
        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.count = 0
        self.video_iterator = 0
        self.frame_iterator = 0
        self.frame_counter = 0
        return


def create_sequence(dir):
    random.seed(35)
    count_real = 0
    count_fake = 0
    folders = [i for i in sorted(glob.glob(os.path.join(dir, '*', "*")))]
    total_folders = len(folders)
    # print(total_files)
    X = {}
    y = []
    print('Total Videos Folders Found:', total_folders)
    pre_folder = -1
    i = 0
    while i < total_folders:
        # for i in range(0,total_files):

        folder = random.choice(folders)
        # print(folder)
        dir_name = os.path.dirname(folder)
        folder_name = os.path.basename(dir_name)

        # print (folder_name)
        if folder_name == 'real':
            if pre_folder == 1:
                continue
            y.append(1)
            count_real += len([i for i in sorted(glob.glob(os.path.join(folder, "*")))])
            pre_folder = 1
        elif folder_name == 'fake':
            if pre_folder == 0:
                continue
            y.append(0)
            count_fake += len([i for i in sorted(glob.glob(os.path.join(folder, "*")))])
            pre_folder = 0
        else:
            print("Directory names should be 'real' for label (1) and 'fake' for label (0)")
            exit(0)
        X[str(i)] = [i for i in sorted(glob.glob(os.path.join(folder, "*")))]
        # print(file)
        folders.remove(folder)
        i += 1
    print('Real:', count_real, 'Fake:', count_fake)
    labels = []
    for i in range(0, count_real):
        labels.append(1)
    for i in range(0, count_fake):
        labels.append(0)
    y_ints = [v.argmax() for v in to_categorical(labels)]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
    print('Class Weights:', class_weights)
    return X, y, class_weights


train_video_per_batch=5
val_video_per_batch=6
test_video_per_batch=4
frames_per_video_per_batch=5
X_train,y_train, class_weights_train =create_sequence('data_256/train/')
X_val,y_val,class_weights_val=create_sequence('data_256/val/')
X_test,y_test,class_weights_test=create_sequence('data_256/test/')
train_it=DFVDSequence(X_train,y_train,train_video_per_batch,frames_per_video_per_batch)
val_it=DFVDSequence(X_val,y_val,val_video_per_batch,frames_per_video_per_batch)
test_it=DFVDSequence(X_test,y_test,test_video_per_batch,frames_per_video_per_batch)

# input_tensor = Input(shape=(None,128, 128, 3))
# model=ResNet(input_shape=(None,128, 128, 3), classes=2, block='bottleneck', residual_unit='v2',
#            repetitions=None, initial_filters=64, activation='softmax', include_top=False,
#            input_tensor=None, dropout=None, transition_dilation_rate=(1, 1),
#            initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max',
#            final_pooling=None, top='classification')
model=ResNet(input_shape=(frames_per_video_per_batch,256, 256, 3), classes=2, block='bottleneck', residual_unit='v2',
           repetitions=None, initial_filters=64, activation='softmax', include_top=False,
           input_tensor=None, dropout=0.25, transition_dilation_rate=(1, 1),
           initial_strides=(2, 2), initial_kernel_size=(7, 7), initial_pooling='max',
           final_pooling=None, top='classification')
print_summary(model, line_length=150, positions=None, print_fn=None)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

_id = dt.now().strftime("%Y-%m-%d_%H.%M.%S")
filepath=_id+"_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(generator=train_it, validation_data=val_it,epochs=50,callbacks=callbacks_list,shuffle=False,class_weight=class_weights_train)



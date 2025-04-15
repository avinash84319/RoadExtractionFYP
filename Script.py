# %%
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Activation,
    BatchNormalization, Conv2DTranspose, Concatenate
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers


# %%
os.environ['CUDA-DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA-VISIBLE_DEVICE'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
!nvidia-smi

# %%
# import kagglehub

# path = kagglehub.dataset_download("balraj98/massachusetts-roads-dataset")

# print("Path to dataset files:", path)

path = "./1"

# %%
train_dir = path+'/tiff/train/'
mask_dir = path+'/tiff/train_labels/'
 
val_dir = path+'/tiff/val/'
v_mask_dir = path+'/tiff/val_labels/'

test_dir = path+'/tiff/test/'
t_mask_dir = path+'/tiff/test_labels/'

image_shape = (256,256)

# %%
def preprocess_mask_image2(image, class_num, color_limit):
  pic = np.array(image)
  img = np.zeros((pic.shape[0], pic.shape[1], 1))  
  np.place(img[ :, :, 0], pic[ :, :, 0] >= color_limit, 1)  
  return img

# %%
def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0) 
 
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# %%
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# %%
def train_generator(img_dir, label_dir, batch_size, input_size):
    list_images = os.listdir(label_dir)
    # shuffle(list_images) #Randomize the choice of batches
    ids_train_split = range(len(list_images))

    while True:
         for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]

            for id in ids_train_batch:
              img_name = img_dir + list_images[id]+'f'
              mask_name = label_dir + list_images[id]
  
              img = cv2.imread(img_name) 
              img  = cv2.resize(img, image_shape, interpolation=cv2.INTER_AREA)
  
              mask = cv2.imread(mask_name)
              mask = cv2.resize(mask, image_shape, interpolation=cv2.INTER_AREA)
              mask = preprocess_mask_image2(mask, 2, 50)                
              
              x_batch += [img]
              y_batch += [mask]    

    
            x_batch = np.array(x_batch) / 255.
            y_batch = np.array(y_batch) 

            yield x_batch, np.expand_dims(y_batch, -1)

# %%
OUTPUT_CHANNELS = 3
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

# %%
image = cv2.imread(path+"/tiff/val/24328810_15.tiff")

plt.imshow(image)

# %%
# Setting parameter values 
t_lower = 250  # Lower Threshold 
t_upper = 300  # Upper threshold 

# Applying the Canny Edge filter 
edge = cv2.Canny(image,t_lower,t_upper)

plt.imshow(edge,cmap='gray')

# %%
from tensorflow.keras.metrics import Precision, Recall, AUC   # Import for IOU calculation
precision = Precision(name='precision')
recall = Recall(name='recall')
iou = AUC(name='iou')
f1 = F1Score(name='f1_score')

accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy') 

import tensorflow as tf
import cv2

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=not apply_batchnorm))
    
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())
    
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())
    
    return result

class LearnableEdgeDetector(tf.keras.layers.Layer):
    def __init__(self, name="learnable_canny", k=50.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k  # controls sharpness of sigmoid
        self.low_threshold = self.add_weight("low_thresh", shape=(), initializer="random_uniform", trainable=True)
        self.high_threshold = self.add_weight("high_thresh", shape=(), initializer="random_uniform", trainable=True)

    def call(self, inputs):
        # Grayscale conversion
        gray = tf.image.rgb_to_grayscale(inputs)
        
        # Compute gradient magnitude
        sobel = tf.image.sobel_edges(gray)
        sobel_x = sobel[..., 0]
        sobel_y = sobel[..., 1]
        magnitude = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))
        
        # Normalize magnitude [0, 1]
        magnitude = magnitude / (tf.reduce_max(magnitude) + 1e-8)
        
        # Differentiable soft thresholding using sigmoid
        strong = tf.sigmoid(self.k * (magnitude - self.high_threshold))
        weak = tf.sigmoid(self.k * (magnitude - self.low_threshold))
        
        # Compute soft edge map: strong + 0.5 * (weak-only)
        edge_map = strong + 0.5 * (weak * (1.0 - strong))
        
        # Ensure values are in [0, 1]
        return tf.clip_by_value(edge_map, 0.0, 1.0)

def Net(num_classes=1, input_shape=(256,256, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    down_stack = [
        downsample(32, 4, apply_batchnorm=False),
        downsample(64, 4),
        downsample(128, 4),
        downsample(256, 4),
    ]

    up_stack = [
        upsample(256, 4, apply_dropout=True),
        upsample(128, 4),
        upsample(64, 4),
        upsample(32, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(num_classes, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='sigmoid')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # ⬇️ Updated edge branch
    edge_branch = LearnableEdgeDetector()(inputs)
    edge_branch = tf.image.resize(edge_branch, (128,128))
    edge_branch = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(edge_branch)

    x = tf.keras.layers.Concatenate()([x, edge_branch])
    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# Instantiate the model
model = Net()
model.summary()


# %%
tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

# %%
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=adam,
                  loss=dice_coef_loss,
                metrics=[precision, recall, iou, accuracy,f1])

# %%
# Define the callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',    # Monitor validation loss
    patience=3,            # Number of epochs to wait before stopping
    restore_best_weights=True,  # Restore the best weights when stopped
    verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',       # Path to save the best model
    monitor='val_loss',    # Monitor validation loss
    save_best_only=True,   # Save the best model based on the metric
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',    # Monitor validation loss
    factor=0.2,            # Factor by which the learning rate will be reduced
    patience=2,            # Number of epochs with no improvement before reducing LR
    verbose=1
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',      # Directory to save TensorBoard logs
    histogram_freq=1,      # Frequency of histogram computation
    write_graph=True,      # Write the computation graph
    write_images=True      # Write images to TensorBoard
)


batch_size = 2
history = model.fit_generator(train_generator(train_dir, mask_dir, batch_size, image_shape),                              
                              steps_per_epoch=554,
                              epochs=10,
                              verbose=1,
                              callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard_callback],
                              validation_data=train_generator(val_dir, v_mask_dir, batch_size, image_shape),
                              validation_steps=1,
                              class_weight=None,
                              max_queue_size=10,
                              workers=1
                              )

# %%
model.save_weights('my_checkpoint')

# %%
train_loss = history.history['loss']
train_recall = history.history['recall']
train_precision = history.history['precision']
train_iou = history.history['iou']

val_loss = history.history['val_loss']
val_recall = history.history['val_recall']
val_precision = history.history['val_precision']
val_iou = history.history['val_iou']

epochs = range(len(train_loss))  # Assuming loss is recorded for each epoch

# Plot loss curves
plt.figure()
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')  # Save the plot as an image

plt.figure()
plt.plot(epochs, train_recall, label='Training Recall')
plt.plot(epochs, val_recall, label='Validation Recall')
plt.title('Training and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.savefig('recall_curve.png')

plt.figure()
plt.plot(epochs, train_precision, label='Training Precision')
plt.plot(epochs, val_precision, label='Validation Precision')
plt.title('Training and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.savefig('precision_curve.png')

plt.figure()
plt.plot(epochs, train_iou, label='Training IOU')
plt.plot(epochs, val_iou, label='Validation IOU')
plt.title('Training and Validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.grid(True)
plt.savefig('iou_curve.png')

plt.show() 

# %%
import pandas as pd

# Extracting history from the model's training process
history_data = history.history

# Convert history data into a DataFrame
history_df = pd.DataFrame({
    'epoch': range(1, len(history_data['loss']) + 1),
    'loss': history_data['loss'],
    'precision': history_data['precision'],
    'recall': history_data['recall'],
    'iou': history_data['iou'],
    'accuracy': history_data['accuracy'],
    'f1': history_data['f1_score'],
    'val_loss': history_data['val_loss'],
    'val_precision': history_data['val_precision'],
    'val_recall': history_data['val_recall'],
    'val_iou': history_data['val_iou'],
    'val_accuracy': history_data['val_accuracy'],
    'val_f1': history_data['val_f1_score']
})

# Display the table
history_df.tail()

# %%
def prepare_test_image(image):    
  x_batch = []   
  # img = cv2.imread(image_path)  
  img  = cv2.resize(image, image_shape, interpolation=cv2.INTER_AREA)
  x_batch += [img]           
  x_batch = np.array(x_batch) / 255.        

  return x_batch



# %%
def binaryImage(image):
  x = image.shape[1]
  y = image.shape[2]
  imgs = np.zeros((x,y,3))
  for k in range(x):
    for n in range(y):
      if image[0,k,n]>0.5:
        imgs[k,n,0]=255
        imgs[k,n,1]=255
        imgs[k,n,2]=255
        # print(imgs[k,n])
      # else:
      #   imgs[k,n]=0
  return imgs 



def draw(orig_im, mask_im, recogn_im, out_im):  
  plt.figure(figsize=(20,17))
  plt.subplot(1,4,1)
  plt.title('Original')
  plt.imshow(orig_im)
  plt.subplot(1,4,2)
  plt.title('Mask Original')
  plt.imshow(mask_im)
  plt.subplot(1,4,3)
  plt.title('Recogn Roads')
  plt.imshow(recogn_im)
  plt.subplot(1,4,4)
  plt.title('Out Unet')
  plt.imshow(out_im)
  plt.axis('off')
  plt.show()




def recogn_test_image():
  test_images = os.listdir(t_mask_dir)

  for test in test_images:
    im_test = cv2.imread(test_dir+test+'f')
    im_mask = cv2.imread(t_mask_dir+test)
    out_test = model.predict(prepare_test_image(im_test), verbose=0)
    img_r = binaryImage(out_test)
    draw(im_test, im_mask, img_r, out_test[0,:, :, 0]*255) 

  


# %%
recogn_test_image()

# %%




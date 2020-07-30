# Module imports

# importing winsound so that it will be triggered after programs executio completes

import winsound

# reducing tensorflow logging to errors only
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

# elementary model layers for seamese model construction
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Dense, Flatten, concatenate, UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.regularizers import l2
import keras.backend as K

# For image loading
from PIL import Image

# for sorting w.r.t siamese distance and saving as csv 
import pandas as pd


# checking tensorflow for GPU execution
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Available GPU : {}".format(gpu_devices))

#  setting GPU memory growth
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)


def initialize_weights(shape, dtype, name=None):
    """ The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01

    Returns:
        random_weights: random weight initializer value
    """    
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape).astype('float')

def initialize_bias(shape, dtype, name=None):
    """ The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01

    Returns:
        random_bias: random bias initializer value
    """    
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape).astype('float32')

def get_siamese(input_shape):
    """ function for siamese model generation

    Args:
        input_shape ([tuple]): input shape of the model
    Returns:
        siamese_model ([tensorflow model instance]): siamese model generated using tensorflow keras layers
    """
    
    # Definnes the tensors for two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Siamese convolutional neural network U-Net
    inputs = Input(shape=(input_shape))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first',
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(conv5)
    #conv5 = Flatten()(conv5)
    model = Model(inputs=inputs, outputs=conv5)
    # upto conv2d_9


    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings of shape (32, 48, 48)
    def diff_encoding(tensors):
        # convert (None, 32, 48, 48) to (None, 1, 48, 48) for input_1
        temp1 = tensors[0][:, 1,: , :] + tensors[0][:, 15,: , :] + tensors[0][:, 21,: , :] + tensors[0][:, 26,: , :] + tensors[0][:, 28,: , :]
        
        # convert (None, 32, 48, 48) to (None, 1, 48, 48) for input_2
        temp2 = tensors[1][:, 1,: , :] + tensors[1][:, 15,: , :] + tensors[1][:, 21,: , :] + tensors[1][:, 26,: , :] + tensors[1][:, 28,: , :]
        
        # convert (None, 1, 48, 48) to (None, 2304) for input_1
        temp1_flatten = K.reshape(temp1, shape=[-1, 48*48])
        # convert (None, 1, 48, 48) to (None, 2304) for input_2
        temp2_flatten = K.reshape(temp2, shape=[-1, 48*48])
        
        # return absolute difference between two tensors
        return K.abs(temp1_flatten - temp2_flatten)    
    
    L1_layer = Lambda(diff_encoding)
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
        
    
    # return the model
    return siamese_net

def get_patches_non_overlap(array, patch_height, patch_width):
    """function to extract non overlapping patches of shape patch_height * patch_width from array

    Args:
        array ([numpy.array]): single dimensonal image array 
        patch_height ([integer]): required non-overlapping height of patches
        patch_width ([integer]): required non-overlapping width of patches

    Returns:
        patches ([numpy.array]): patch array with shape=(total_patches, 1, patch_height, patch_width)
    """    
    total_patches_in_height = array.shape[0]//patch_height
    total_patches_in_width = array.shape[1]//patch_width
    print("total patches in height from supplied image array : {}".format(total_patches_in_height))
    print("total patches in width from supplied image array : {}".format(total_patches_in_width))
    
    total_patches = total_patches_in_height * total_patches_in_width
    print("total patches from supplied image array : {}".format(total_patches))
    patches = np.empty(shape=(total_patches, 1, patch_height, patch_width), dtype=np.uint8)
    
    patch_no = 0
    for i in range(0, array.shape[0], patch_height):
        for j in range(0, array.shape[1], patch_width):
            if (i+patch_height <= array.shape[0]+1) and (j+patch_width <= array.shape[1]+1):
                patches[patch_no, 0, :, :] = array[i:i+patch_height, j:j+patch_width]
                patch_no += 1
    return patches
    

def compare(model, input1, input2):
    """function to compare input1 and input2 using siamese model

    Args:
        model ([tensorflow.keras.models]): model instance created using tensorflow.keras layers
        input1 ([numpy.array]): image in numpy array form
        input2 ([numpy.array]): image in numpy array form

    Returns:
        np.sum(pred) ([float32]): sum of siamese distance returned by model.predict
    """    
    input1_patches = get_patches_non_overlap(input1, 48, 48)
    input2_patches = get_patches_non_overlap(input2, 48, 48)
    pred = model.predict([input1_patches, input2_patches])
    return np.sum(pred)

def driver(rootdir):
    """driver program for OCT image retrieval using siamese net and saves the retieval result in csv file

    Args:
        rootdir ([string]): dataset directory

    Returns:
        [type]: [description]
    """    
    siamese_model = get_siamese(input_shape=(1, 48, 48))
    siamese_model.summary()
    
    for subdir1, dirs1, files1 in os.walk(rootdir):
        destination = "..\\result\\seamese_unet\\" # + subdir1.split("\\")[-1]
        query1  = subdir1.split("\\")[-1]
        
        os.makedirs(destination, exist_ok=True)
        
        result = {"query1": [], "query2":[], "size": [], "siamese_distance": []}
        
        if not subdir1.endswith("\\Duke-selected-new\\"):
            for subdir2, dirs2, files2 in os.walk(rootdir):
                if not subdir2.endswith("\\Duke-selected-new\\"):
                    if (subdir1 != subdir2):
                        query2  = subdir2.split("\\")[-1]
                        print(subdir1, subdir2)
                        img1_path = os.path.join(subdir1, "01.tif")
                        print(img1_path)
                        img1 = np.asarray(Image.open(img1_path))
                        
                        img2_path = os.path.join(subdir2, "01.tif")
                        print(img2_path)
                        img2 = np.asarray(Image.open(img2_path))
                        
                        siamese_distance = compare(siamese_model, img1, img2)
                        print("siamese_distance between {} and {} value : {}".format(query1, query2, siamese_distance))
                        
                        result["query1"].append(query1)
                        result["query2"].append(query2)
                        result["size"].append(img1.shape)
                        result["siamese_distance"].append(siamese_distance)
        
        #save result tp csv file sorted w.r.t siamese_distance
            df = pd.DataFrame(data=result)
            df = df.sort_values(by=["siamese_distance"])
            df.to_csv(destination + "\\" + query1 +".csv")
    
if __name__ == "__main__":
    driver("H:\\OCT retrieval\\Dataset\\Duke-selected-new\\")

    # reporting end of program execution by beep sound
    winsound.Beep(2500, 4000)

# Module imports

# importing winsound so that it will be triggered after programs executio completes

import winsound

# reducing tensorflow logging to errors only
import os

from numpy.core.fromnumeric import shape
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

# elementary model layers for seamese model construction
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.random import set_seed
import keras.backend as K


# For image loading
from PIL import Image

# for sorting w.r.t siamese distance and saving as csv 
import pandas as pd


# setting seed for numpy module
np.random.seed(2)

# setting seed for tensoflow module
set_seed(2)


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
    
    # Seamese convolutional neural network
    model = Sequential()
    model.add(Conv2D(64, (10,10), padding='same', data_format='channels_first', activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), padding='same', data_format='channels_first', activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), padding='same', data_format='channels_first', activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), padding='same', data_format='channels_first', activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
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
    # print("total patches in height from supplied image array : {}".format(total_patches_in_height))
    # print("total patches in width from supplied image array : {}".format(total_patches_in_width))
    
    total_patches = total_patches_in_height * total_patches_in_width
    # print("total patches from supplied image array : {}".format(total_patches))
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


def averageImage(dir):
    for subdir, dirs, files in os.walk(dir):
        # print(subdir, files)
        try:
            total_image = np.zeros(shape=(496, 512))
            for file in files:
                img_path = os.path.join(subdir, file)
                total_image = np.add(total_image, np.asarray(Image.open(img_path)))
        except:
            total_image = np.zeros(shape=(496, 768))
            for file in files:
                img_path = os.path.join(subdir, file)
                total_image = np.add(total_image, np.asarray(Image.open(img_path)))
                
    if total_image.shape == (496, 768):
        total_image = total_image[:, 128:(768-127)]
    return total_image/len(files)
        
def calculateAvgPrecision(df, k):
    total_true = 0
    for i in range(k):
        print(df['query1'].iloc[0], df['query2'].iloc[i])
        if df['query1'].iloc[0][0] == df['query2'].iloc[i][0]:
            total_true += 1
    print(df["query1"][0], total_true)
    return total_true/k         

def calculateReciprocalRank(df, k):
    total_true = 0
    for i in range(k):
        if df['query1'].iloc[0][0] == df['query2'].iloc[i][0]:
            return 1/(i+1)
    return 0

def driver(rootdir, destination):
    """driver program for OCT image retrieval using siamese net and saves the retieval result in csv file

    Args:
        rootdir ([string]): dataset directory
        destination ([string]): result storage directory
    Returns:
        [type]: [description]
    """
    
    metric_result = {"query image": [], "k": [], "average precision": [], "reciprocal rank": []}
    
    siamese_model = get_siamese(input_shape=(1, 48, 48))
    siamese_model.summary()
    APlist = []
    RRlist = []
    # destination = "..\\result\\seamese_net_avg_images_seed_np_2_tf_2\\" # + subdir1.split("\\")[-1]
    for subdir1, dirs1, files1 in os.walk(rootdir):

        query1_name  = subdir1.split("\\")[-1]
        
        os.makedirs(destination, exist_ok=True)
        
        query1 = averageImage(subdir1)
        
        result = {"query1": [], "query2":[], "size": [], "siamese_distance": []}
        
        
        if not subdir1.endswith("\\Duke-DME-Normal\\"):
            for subdir2, dirs2, files2 in os.walk(rootdir):
                if not subdir2.endswith("\\Duke-DME-Normal\\"):
                    if (subdir1 != subdir2):
                        query2_name  = subdir2.split("\\")[-1]
                        # print(subdir1, subdir2)
                        
                        query2 = averageImage(subdir2)
                        
                        siamese_distance = compare(siamese_model, query1, query2)
                        # print("siamese_distance between {} and {} value : {}".format(query1_name, query2_name, siamese_distance))
                        
                        result["query1"].append(query1_name)
                        result["query2"].append(query2_name)
                        result["size"].append((496, 512))
                        result["siamese_distance"].append(siamese_distance)
        
        #save result tp csv file sorted w.r.t siamese_distance
            df = pd.DataFrame(data=result)
            df = df.sort_values(by=["siamese_distance"])
            df.to_csv(destination + "\\" + query1_name +".csv")
            
            APlist.append(calculateAvgPrecision(df, 3))
            RRlist.append(calculateReciprocalRank(df, 3))
            # print(APlist, RRlist)
            metric_result["query image"].append(query1_name)
            metric_result["k"].append(3)
            metric_result["average precision"].append(calculateAvgPrecision(df, 3))
            metric_result["reciprocal rank"].append(calculateReciprocalRank(df, 3))
            
            
    print("Mean Average Precision (MAP) considering K = 3 : {}".format(sum(APlist)/len(APlist)))
    print("Mean Reciprocal Rank (MRR) considering K = 3 : {}".format(sum(RRlist)/len(RRlist)))
    
    metric_result["query image"].append("Average MAP and MRR")
    metric_result["k"].append(3)
    metric_result["average precision"].append(sum(APlist)/len(APlist))
    metric_result["reciprocal rank"].append(sum(RRlist)/len(RRlist))
    metric_df = pd.DataFrame(data=metric_result)
    metric_df.to_csv(destination + "\\" + "CBIR metric.csv")
    
if __name__ == "__main__":
    driver("J:\\OCT retrieval\\Dataset\\Duke-DME-Normal\\", "..\\result\\seamese_net_avg_images_seed_np_2_tf_2\\")

    # reporting end of program execution by beep sound
    winsound.Beep(2500, 4000)

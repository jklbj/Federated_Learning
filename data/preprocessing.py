import cv2
import numpy as np
from sklearn.utils import shuffle
from random import randint
from keras.utils import to_categorical
import random

def random_contrast(im, lower=0.2, upper=1.8):
    """
    It will randomly adjust contrast of input image.
    :param image to be adjusted
    :param contrast lower range (default 0.2)
    :param contrast higher range (default 1.8)
    :return processed image
    """
    prob = randint(0, 1)
    if prob == 1:
        alpha = random.uniform(lower, upper)
        imgg = im * alpha
        imgg = imgg.clip(min=0, max=1)
        return imgg
    else:
        return im

def random_bright(im, delta=63):
    """
    It will randomly adjust brightness of input image.
    :param image to be adjusted
    :param brightness range (default range from -63 to 63)
    :return processed image
    """
    prob = randint(0,1)
    if prob == 1:
        delta = random.uniform(-delta, delta)
        imgg = im + delta / 255.0
        imgg = imgg.clip(min=0, max=1)
        return imgg
    else:
        return im

def per_image_standardization(img):
    """
    It will adjust standardization of input image.
    :param image to be adjusted
    :return processed image
    """
    num_compare = img.shape[0] * img.shape[1] * 3
    img_arr=np.array(img)
    img_t = (img_arr - np.mean(img_arr))/max(np.std(img_arr), 1/num_compare)
    return img_t

def random_crop(img, width, height):
    """
    It will randomly crop input image with width and height.
    :param image to be adjusted
    :param crop width
    :param crop height
    :return processed image
    """
    width1 = randint(0, img.shape[0] - width)
    height1 = randint(0, img.shape[1] - height)
    cropped = img[height1:height1+height, width1:width1+width]

    return cropped

def random_flip_left_right(image):
    """
    It will randomly filp the input image left or right.
    :param image to be flipped
    :return processed image
    """
    prob = randint(0, 1)
    if prob == 1:
        image = np.fliplr(image)
    return image

def preprocessing_for_training(images):
    """
    It will preprocess the input image for training.
    :param image for training to be proprecessed in each epoch
    :return proprecessed image for training
    """
    distorted_image = random_flip_left_right(images)
    distorted_image = random_bright(distorted_image)
    distorted_image = random_contrast(distorted_image)
    float_image = per_image_standardization(distorted_image)
    
    return float_image

def preprocessing_for_testing(images):
    distorted_image = cv2.resize(images, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
    distorted_image = per_image_standardization(distorted_image)
    
    return distorted_image

def separate_and_preprocess_for_simple_fed(device_num, train_images, train_labels, num_device):
    """
    It will seperate and preprocess (i.e., randomly crop each image for training) 
    the training images for the target device.
    :param training images to be seperated to the target device
    :param training labels to be seperated to the target device
    :param data distribution for all devices
    :param device number which means which device data to be seperated and preprocessed 
    :return the target device data (i.e., training data, training labels) have been seperated and preprocessed
    """
    train_image_temp, train_label_temp = prepare_for_training_data(device_num, train_images, train_labels, num_device)
    train_image_crop = np.stack([random_crop(train_image_temp[i], 24, 24) for i in range(len(train_image_temp))], axis=0)
    
    # Shuffle for 20 times
    for random in range(20):
        train_image_crop, train_label_temp = shuffle(train_image_crop, 
                                            train_label_temp, 
                                            random_state=randint(0, train_image_crop.shape[0]))
    return train_image_crop, train_label_temp

def separate_and_preprocess_for_fed_plus(train_images, train_labels, data_distribution, device):
    """
    It will seperate and preprocess (i.e., randomly crop each image for training) 
    the training images for the target device.
    :param training images to be seperated to the target device
    :param training labels to be seperated to the target device
    :param data distribution for all devices
    :param device number which means which device data to be seperated and preprocessed 
    :return the target device data (i.e., training data, training labels) have been seperated and preprocessed
    """
    train_image_temp, train_label_temp = prepare_for_training_data0(train_images, train_labels, data_distribution, device)
    train_image_crop = np.stack([random_crop(train_image_temp[i], 24, 24) for i in range(len(train_image_temp))], axis=0)
    
    # Shuffle for 20 times
    for random in range(20):
        train_image_crop, train_label_temp = shuffle(train_image_crop, 
                                            train_label_temp, 
                                            random_state=randint(0, train_image_crop.shape[0]))
    return train_image_crop, train_label_temp

# Seperate data for each device
def prepare_for_training_data0(train_images, train_labels, data_distribution, device_num, num_class=10, quantity_of_each_class=5000):
    # Return
    image, label = train_images, train_labels
    all_class_device = data_distribution[device_num]

    device_num_start = data_distribution[device_num]['data_distribution'][0][0] % quantity_of_each_class
    device_num_end = data_distribution[device_num]['data_distribution'][0][1] % quantity_of_each_class

    if device_num_start > device_num_end: 
        a = image[label[:, 0] == 0][device_num_start:]
        b = image[label[:, 0] == 0][:device_num_end]

        a_label = label[label[:, 0] == 0][device_num_start:]
        b_label = label[label[:, 0] == 0][:device_num_end]
        s0 = [np.vstack((a,b)), np.vstack((a_label,b_label))]
    else:
        s0 = [image[label[:, 0] == 0][device_num_start : device_num_end], label[label[:, 0] == 0][device_num_start : device_num_end]]

    for i in range(1, num_class):
        device_num_start = data_distribution[device_num]['data_distribution'][0][0] % 5000
        device_num_end = data_distribution[device_num]['data_distribution'][0][1] % 5000

        if device_num_start > device_num_end: 
            a = image[label[:, 0] == i][device_num_start:]
            b = image[label[:, 0] == i][:device_num_end]

            a_label = label[label[:, 0] == i][device_num_start:]
            b_label = label[label[:, 0] == i][:device_num_end]
            s1 = [np.vstack((a,b)), np.vstack((a_label,b_label))]
        else:
            s1 = [image[label[:, 0] == i][device_num_start : device_num_end], label[label[:, 0] == i][device_num_start : device_num_end]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]


    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1])


def prepare_for_training_data(device_num, train_images, train_labels, num_device):
    num_data = int(len(train_images)/num_device/10)
    device_num = device_num * num_data
    
    image, label = train_images, train_labels
    
    s0 = [image[label[:, 0] == 0][device_num : device_num+num_data], label[label[:, 0] == 0][device_num : device_num+num_data]]

    for i in range(1, 10):
        s1 = [image[label[:, 0] == i][device_num : device_num+num_data], label[label[:, 0] == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1])

def evaluate_with_new_model(round, training_info, model_x, test_images, test_label):
    """
    It will evaluate the new model weight which is aggregrated from all client model.
    :param what round is now
    :param dictionary contains training detailed settings
    :param the model for evaluation
    :param original images for testing
    :param original labels for testing
    :return the evaluation result of the input model
    """
    # Prepare for images and labels for evaluating new model weight
    test_new_image, test_new_label = prepare_for_evaluate(test_images, test_label)
    # Evaluate with new weight
    history_temp = model_x.evaluate(test_new_image, 
                                    test_new_label, 
                                    batch_size=training_info["center_batch_size"],
                                    verbose=training_info["show"])
    print("\nRound", str(round), "\nAccuracy:", str(history_temp[1]) + ", Loss:", str(history_temp[0]))

    return history_temp

def prepare_for_evaluate(test_images, test_label):
    """
    It will preprocess and return the images and labels for tesing.
    :param original images for testing
    :param original labels for testing
    :return preprocessed images
    :return preprocessed labels
    """
    test_d = np.stack([preprocessing_for_testing(test_images[i]) for i in range(10000)], axis=0)
    test_new_image, test_new_label = test_d, test_label
    
    # Shuffle for 20 times
    for time in range(20):
        test_new_image, test_new_label = shuffle(test_d, test_label, 
                                             random_state=randint(0, test_images.shape[0]))
    return test_new_image, test_new_label

    

def prepare_for_testing_data(test_images, test_labels, device_num, num_device, num_class=10):
    num_data = int(len(test_images)/num_device/num_class)
    device_num = device_num * num_data

    image, label = test_images, test_labels
#     image, label = shuffle(test_images, test_labels, random_state=0)
    
    s0 = [image[label[:, 0] == 0][device_num : device_num+num_data], label[label[:, 0] == 0][device_num : device_num+num_data]]

    for i in range(1, num_class):
        s1 = [image[label[:, 0] == i][device_num : device_num+num_data], label[label[:, 0] == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1])

def prepare_for_training_data_mnist(device_num, train_images, train_labels, num_device):
    num_data = int(len(train_images)/num_device/10)
    device_num = device_num * num_data

    image, label = train_images, train_labels
    
    s0 = [image[label == 0][device_num : device_num + num_data], label[label == 0][device_num : device_num + num_data]]

    for i in range(1, 10):
        s1 = [image[label == i][device_num : device_num+num_data], label[label == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1], 10)


def prepare_for_testing_data_mnist(device_num, test_images, test_labels, num_device):
    num_data = int(len(test_images)/num_device/10)
    device_num = device_num * num_data

    image, label = test_images, test_labels
#     image, label = shuffle(test_images, test_labels, random_state=0)
    
    s0 = [image[label == 0][device_num : device_num+num_data], label[label == 0][device_num : device_num+num_data]]

    for i in range(1, 10):
        s1 = [image[label == i][device_num : device_num+num_data], label[label == i][device_num : device_num+num_data]]

        s0 = [np.concatenate((s0[0], s1[0]), axis=0), np.append(s0[1], s1[1])]

    s0[0], s0[1] = shuffle(s0[0], s0[1], random_state=randint(0, device_num))
    
    return s0[0], to_categorical(s0[1], 10)

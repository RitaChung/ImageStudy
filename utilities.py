import cv2
import tensorflow.keras
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.svm import SVC


class ImageReader():
    def __init__(self, path, resize_dim=28):
        self.path = path
        self.resize_dim

    def toImage(self):
        img = cv2.imread(self.path)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img

    def toResizeImage(self):
        img = cv2.imread(self.path)
        img = cv2.resize(img, (self.resize_dim, self.resize_dim))
        img = img / 255.0
        return img



class Autoencoder():
    def __init__(self, original_dim=784, encoding_dim=32, input_dim = 28, output_dim=80000):
        self.encoding_dim = encoding_dim
        self.original_dim = original_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

    def autoencoder(self):
        input_img = Input(shape=(self.original_dim,self.encoding_dim))
        encoded = Dense(self.encoding_dim, activation='relu')(input_img)
        decoded = Dense(self.original_dim, activation='sigmoid')(encoded)
        return Model(input_img, decoded)

    def conv_autoencoder(self):
        input_img = Input(shape=(self.input_dim, self.input_dim,3))
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
        conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
        encoded = MaxPooling2D((2, 2), padding='same')(conv3)

        #decoder
        conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        up1 = UpSampling2D((2, 2))(conv4)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
        up2 = UpSampling2D((2, 2))(conv5)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
        up3 = UpSampling2D((2, 2))(conv6)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)
        return Model(input_img, decoded)

    def InceptionV3(self):
        input_img = Input(shape=(self.input_dim, self.input_dim,3))
        base_model = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=True)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.output_dim, activation='softmax')(x)
        return Model(base_model.input, predictions)

class Classifier():
    def __init__(self, kernal='rbf'):
        self.kernal = kernal

    def svm(self):
        svclassifier = SVC(kernel=self.kernal)
        return svclassifier

class local_features_extractor():
    def __init__(self, path):
        self.name = path

    def toImage(self):
        img = cv2.imread(self.name)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def feature(self):
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.toImage(), None)
        collect = dict()
        collect['name'] = self.name
        collect['num_of_keypoints'] = len(keypoints)
        collect['keypoints'] = keypoints
        collect['descriptors'] = descriptors
        return collect

class imageMatcher():
    def __init__(self, collect1, collect2, valid_distance = 100000):
        self.obj1 = collect1
        self.obj1_num_of_kps = collect1['num_of_keypoints']
        self.obj1_keypoint = collect1['keypoints']
        self.obj1_descriptor = collect1['descriptors']

        self.obj2 = collect2
        self.obj2_num_of_kps = collect2['num_of_keypoints']
        self.obj2_keypoint = collect2['keypoints']
        self.obj2_descriptor = collect2['descriptors']

        self.valid_distance = valid_distance

    def matcher(self):
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
        matches = bf.match(self.obj1_descriptor, self.obj2_descriptor)
        new_matchers = [matches[run] for run in range(len(matches)) if matches[run].distance < self.valid_distance]
        new_matchers = sorted(new_matchers, key=lambda x: x.distance)
        return new_matchers

    def similarity(self):
        stat = dict()
        stat['num_of_valid_matches'] = len(self.matcher())
        stat['num_of_compare'] = self.obj1_num_of_kps * self.obj2_num_of_kps
        stat['percent_of_matches'] = len(self.matcher()) / (self.obj1_num_of_kps * self.obj2_num_of_kps) * 100
        return stat

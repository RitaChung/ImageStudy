import os
import glob
import pandas as pd
from utilities import *
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

source_path = '/raid/rita/lr'
train_source_path = os.path.join(source_path,'train')
test_source_path = os.path.join(source_path,'test')

#train
train_label = pd.read_csv(os.path.join(source_path,'train.csv'), header=0)
train_label['root_path'] = train_source_path
train_label['p1'] = train_label['id'].str.get(0)
train_label['p2'] = train_label['id'].str.get(1)
train_label['p3'] = train_label['id'].str.get(2)
train_label['path'] = train_label['root_path'].str.cat(train_label['p1'], sep="/").str.cat(train_label['p2'], sep="/").str.cat(train_label['p3'], sep="/").str.cat(train_label['id'], sep="/") + '.jpg'
train_label['landmark_id'] = train_label['landmark_id'].astype('str')
#test
test_path = pd.DataFrame({'id':[], 'path':[]})
for path in glob.glob(os.path.join(test_source_path,'*/*/*/*')):
    word = path.split('/')
    id = word[len(word)-1].replace('.jpg','')
    test_path = test_path.append(pd.DataFrame([[id, path]], columns = ['id','path']), ignore_index = True)

#parameter
input_dim = 128
output_dim = len(np.unique(train_label['landmark_id']))
batch_size = 128
epoch = 1

##train autoencoder
dataGen = ImageDataGenerator(rescale=1./255)
train_for_train_dataGen = dataGen.flow_from_dataframe(dataframe=train_label, x_col='path', y_col='landmark_id',
                                                        target_size=(input_dim, input_dim), color_mode="rgb",
                                                        class_mode="input", batch_size=batch_size)
test_for_predict_dataGen = dataGen.flow_from_dataframe(dataframe=test_path, x_col='path', y_col='id',
                                                        target_size=(input_dim, input_dim), color_mode="rgb",
                                                        class_mode=None, batch_size=1)
step_size_train = train_for_train_dataGen.n//train_for_train_dataGen.batch_size

with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    modelGenerator = Autoencoder(input_dim=input_dim,output_dim=output_dim)
    convautoencoder = modelGenerator.conv_autoencoder()
    convautoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    convautoencoder.fit_generator(generator=train_for_train_dataGen, steps_per_epoch = step_size_train, epochs=epoch)
    intermediate_output = convautoencoder.get_layer('conv2d_4').output
    encoder = Model(convautoencoder.input, intermediate_output)
    test_autoencoder_features = encoder.predict_generator(test_for_predict_dataGen, steps=len(test_path))
    print(type(test_autoencoder_features))
    print(test_autoencoder_features.shape)
    print(test_autoencoder_features.head(10))
#train_embedding_features = encoder.predict(x_train)
#test_embedding_features = encoder.predict(x_test)

#train_embedding_features = np.reshape(train_embedding_features, (train_embedding_features.shape[0], train_embedding_features.shape[1]*train_embedding_features.shape[2]*train_embedding_features.shape[3]))
#test_embedding_features = np.reshape(test_embedding_features, (test_embedding_features.shape[0], test_embedding_features.shape[1]*test_embedding_features.shape[2]*test_embedding_features.shape[3]))



#path_list = pd.DataFrame({'source':[], 'id':[], 'path':[]})

#for type in ['train','test']:
#    data_source = os.path.join(source_path,type)
#    print(data_source)
#    for path in glob.glob(data_source + '/*/*/*/*'):
#        word = path.split('/')
#        id = word[len(word)-1].replace('.jpg','')
#        path_list = path_list.append(pd.DataFrame([[type,id, path]], columns = ['source','id','path']), ignore_index = True)

#path_list.to_csv('id_with_path.csv',index=False)

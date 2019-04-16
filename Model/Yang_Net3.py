from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
from scipy.io import loadmat
from datetime import datetime
import tensorflow as tf
import collections
from sklearn import preprocessing
import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def normalize(v):
    data =  (v-v.mean(axis = 1).reshape((v.shape[0], 1))) / (v.max(axis=1).reshape((v.shape[0], 1)) + 2e-12)
    
    return data

def get_feature(wav_file):
    mat = loadmat(wav_file)
    dat = mat["data"]
    feature = dat[0:12]
    return (normalize(feature).transpose())
    

def get_traindata(path):
    
    Window_Length =10*500
    NO_tests =600
    NO_channels = 12
    data_return = np.zeros((NO_tests, Window_Length, NO_channels))
    
    files = []
    count = 0
    for file in os.listdir(path):
        #print(file)
        files.append(file)

    for f in files:
        dat = get_feature(path+f)
        data_return[count,:,:] = dat
        count += 1
        
    print(data_return.shape)
    
    return data_return


def get_testdata(path):

    Window_Length =10*500
    NO_tests =400
    NO_channels = 12
    data_return = np.zeros((NO_tests, Window_Length, NO_channels))
    
    files = []
    count = 0
    for file in os.listdir(path):
        #print(file)
        files.append(file)

    for f in files:
        dat = get_feature(path+f)
        data_return[count,:,:] = dat
        count += 1
        
    print(data_return.shape)
    
    return files, data_return

def load_label(label_path):
    
    label = np.loadtxt(label_path,dtype=str)
    label = label[:,1].astype(np.int32)
    label = np.reshape(label,[-1,1])
    enc_2 = preprocessing.OneHotEncoder() 
    enc_2.fit(label)  
    train_labels = enc_2.transform(label).toarray() 
    labels=np.reshape(train_labels,[-1,2])   
    return labels
 


class DataSet(object):

    def __init__(self,
                 data,
                 labels,
                 one_hot=False,
                 dtype=np.float32,
                 reshape=True):
            
           
        self._num_examples=data.shape[0]
            #print(self._num_examples)
        if reshape:
            #assert data.shape[2] ==1
            data = data.reshape (data.shape[0],
                                      data.shape[1])
        if dtype == np.float32:
            data = data.astype(np.float32)
        self._data = data
        self._labels= labels
        self._epochs_completed =0
        self._index_in_epoch =0
        
    @property
    def data(self):
        return self._data
            
    @property
    def labels(self):
        return self._labels
            
    @property
    def num_exanples(self):
        return self._num_examples
            
    @property
    def epochs_completed(self):
        return self._epochs_completed
        
        
    def next_batch(self, batch_size, shuffle =True):
        """ Return the bext 'batch_size' examples from this data set."""
        start=self._index_in_epoch
                
        #Shuffle for the first epoch
                
        if self._epochs_completed ==0 and start ==0 and shuffle:
            perm0=np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data=self.data[perm0]
            self._labels=self.labels[perm0]
                    
        #go to the next epoch
        if start+batch_size > self._num_examples:
            #finished epoch
            self._epochs_completed +=1
            #get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start : self._num_examples]
            labels_rest_part =self._labels[start: self._num_examples]
            #Shuffle the data      
            if shuffle:
                perm=np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data=self.data[perm]
                self._labels=self.labels[perm]
                        
            # Start next epoch
            start = 0
            self._index_in_epoch =batch_size-rest_num_examples
            end=self._index_in_epoch
            data_new_part = self._data[start: end]
            labels_new_part = self._labels[start: end]
            return np.concatenate((data_rest_part, data_new_part), axis=0),np.concatenate((labels_rest_part, labels_new_part), axis=0)                    
        else:
            self._index_in_epoch += batch_size
            end=self._index_in_epoch
            return self._data[start: end], self._labels[start: end]



        
        
Train_path = 'E:/CorrelationModel/ECG_Competition/TRAIN/'

test_path = 'E:/CorrelationModel/ECG_Competition/TEST/'  

label_path = 'E:/CorrelationModel/ECG_Competition/reference.txt'   
        
Datasets = collections.namedtuple('Datasets', ['train'])

def read_data_sets(dtype = np.float32,
                    reshape = False):
    
    train_data = get_traindata(Train_path)
    train_labels = load_label(label_path)
    train =DataSet(train_data, train_labels, dtype =dtype, reshape =reshape)
    
    return Datasets(train=train)
    


#Convolutional Neural Network Model

def Model(inputs):
  
    # first two layers      
    conv_1=tf.layers.conv1d(inputs,
                            32,
                            kernel_size = 8,
                            padding='same')
    
    norm_1 = tf.layers.batch_normalization(conv_1)

    conv = tf.nn.tanh(norm_1)
        
    conv_1 = tf.layers.conv1d(conv,
                              32,
                                kernel_size=8,
                                padding="same")
            
    norm_1 = tf.layers.batch_normalization(conv_1)

    conv =  tf.nn.tanh(norm_1)

        
    pool = tf.layers.max_pooling1d(inputs=conv, pool_size= 2, strides=2) 


    # Stacking Layer2
    conv_2= tf.layers.conv1d(pool,
                            64,
                            kernel_size=6,
                            padding="same")
            
    norm_2 = tf.layers.batch_normalization(conv_2)
        
    conv =  tf.nn.tanh(norm_2)
        
    conv_2= tf.layers.conv1d(conv,
                            64,
                            kernel_size=6,
                            padding="same")
            
    norm_2 = tf.layers.batch_normalization(conv_2)
        
    conv =  tf.nn.tanh(norm_2)
    
    pool = tf.layers.max_pooling1d(inputs=conv, pool_size= 2, strides=2) 
    
    #Stacking Layer3
    conv_3 = tf.layers.conv1d(pool,
                            128,
                            kernel_size=4,
                            padding="same")
            
    norm_3 = tf.layers.batch_normalization(conv_3)
        
    conv =  tf.nn.tanh(norm_3)
        
    conv_3= tf.layers.conv1d(conv,
                            128,
                            kernel_size=4,
                            padding="same")
            
    norm_3 = tf.layers.batch_normalization(conv_3)
        
    conv =  tf.nn.tanh(norm_3)
    
    pool = tf.layers.max_pooling1d(inputs=conv, pool_size= 2, strides=2)
    
    #Stacking Layer4
    conv_4= tf.layers.conv1d(pool,
                            256,
                            kernel_size=2,
                            padding="same")
            
    norm_4 = tf.layers.batch_normalization(conv_4)
        
    conv =  tf.nn.tanh(norm_4)
        
    conv_4= tf.layers.conv1d(conv,
                            256,
                            kernel_size=2,
                            padding="valid")
            
    norm_4 = tf.layers.batch_normalization(conv_4)
        
    conv =  tf.nn.tanh(norm_4)
    
    
    conv_4= tf.layers.conv1d(conv,
                            256,
                            kernel_size=1,
                            padding="valid")
            
    norm_4 = tf.layers.batch_normalization(conv_4)
        
    conv =  tf.nn.tanh(norm_4)
    
    
    conv_4= tf.layers.conv1d(conv,
                            256,
                            kernel_size=1,
                            padding="valid")
            
    norm_4 = tf.layers.batch_normalization(conv_4)
        
    conv =  tf.nn.tanh(norm_4)
    
    pool = tf.layers.max_pooling1d(inputs=conv, pool_size= 2, strides=2)
        
    return pool



class CNN:
    def __init__(self,alpha,batch_size,num_classes,num_features, num_channels):
        """ Initialize the CNN model
        :param alpha: the learning rate to be used by the model
        :param batch_size : the number of batches to use for training
        :param num_classes: the number of classes in the dataset
        :param num_features: the number of features in the dataset
        """
        
        self.alpha= alpha
        self.batch_size = batch_size
        self.name='CNN'
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_channels = num_channels
        
        def __graph__():
        
            #[batch_size, num_features]
            x_input = tf.placeholder(dtype=tf.float32,shape=[None, num_features, num_channels], name='x_input')
            
            #[batch_size, num_classes*num_labels]
            y_input=tf.placeholder(dtype= tf.float32, shape=[None,num_classes], name='actual_label')
            
            input_layer = tf.reshape(x_input,[-1,5000,12])
            
   
            #Model   
            conv = Model(input_layer)

            drop = tf.layers.dropout(inputs=conv, rate=0.3) 
            #flatten abstract feature
            
            flat_1 = tf.reshape(drop,[-1, 256*312])
            dense = tf.layers.dense(flat_1, units=1024, activation = tf.nn.tanh)
            dense = tf.layers.dense(dense, units=512, activation = tf.nn.tanh)
            #classification 
            digit1 = tf.layers.dense(dense, units=2)
            
            # softmax layer 
            digit = tf.nn.softmax(digit1)
            
            loss_2 =  tf.reduce_mean(-y_input*tf.log(digit+1e-8))
            
            # digit1_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_input, digit1))
           
            l2 =0.0005*sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            #loss function
           
            loss = loss_2 + l2
            tf.summary.scalar('loss',loss)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)
            
            #accuracy
            output=tf.argmax( digit,1)
            
            label = tf.argmax(y_input,1)
            
            correct_pred= tf.equal(label,output)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
         
            tf.summary.scalar('accuracy', accuracy)
         
          
            merged = tf.summary.merge_all()
            
            self.x_input = x_input
            
            self.y_input  = y_input
            self.digit1=digit1
            self.digit2 = digit
            self.loss = loss
            self.output = output
            self.label = label
            self.optimizer=optimizer
            self.accuracy = accuracy
            self.merged = merged
            
        sys.stdout.write('\n<log> Building graph...')
        __graph__()
        sys.stdout.write('</log>\n')
 
        

    def train(self,checkpoint_path, epochs,log_path, train_data):
        
        """
        Trains the initialized model.
        :param checkpoint_path: The path where to save the trained model.
        :param log_path: The path where to save the TensorBoard logs.
        :param train_data: The training dataset.
        :param test_data: The testing dataset.
        :return: None
        """
        
        if not os.path.exists(path=log_path):
            os.mkdir(log_path)
            
        if not os.path.exists(path=checkpoint_path):
            os.mkdir(checkpoint_path)
            
        
        saver= tf.train.Saver(max_to_keep=4)
        
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        #init = tf.global_variables_initializer()
        
        timestamp = str(time.asctime())
            
        train_writer = tf.summary.FileWriter(logdir=log_path +'-training', graph=tf.get_default_graph())
        
        with tf.Session() as sess:
            sess.run(init)
            
            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
            
            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                
            for index in range(epochs): 
                #train by batch
                batch_features, batch_labels = train_data.next_batch(self.batch_size)
                
                #input dictionary with dropout of 50%
                feed_dict = {self.x_input:batch_features, self.y_input:batch_labels}
                
                # run the train op
                summary, _, loss = sess.run([self.merged, self.optimizer, self.loss], feed_dict=feed_dict)
                
                if index % 100 ==0:
                    feed_dict = {self.x_input: batch_features, self.y_input: batch_labels}
                    # get the accuracy of training
                    train_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
                    
                    #dispaly the training accuracy
                    print('step: {}, training accuracy : {}, training loss : {}'.format(index, train_accuracy, loss))
                    
                    train_writer.add_summary(summary=summary, global_step=index)
                    saver.save(sess, save_path=os.path.join(checkpoint_path, self.name), global_step=index)
            
            files,test_data = get_testdata(test_path)
            
            test = []
            for f in files:
                test.append(f.strip().strip('.mat'))
        
            test = np.reshape(test, [-1,1])
            print(test)
        
            feed_dict= {self.x_input:test_data} 
               
            predict = sess.run(self.output, feed_dict=feed_dict)
            predict = np.reshape(predict, [-1, 1])
            
            Test = np.concatenate((test, predict), axis=1)
        
           
            np.savetxt('answers.txt',Test, fmt='%s',delimiter=' ',newline='\n')
            
       
if __name__ == '__main__':
    
    
    data=read_data_sets()
    
    train_data=data.train
    num_classes=2      
    num_features = 5000
    num_channels = 12
    model=CNN(alpha= 0.002, batch_size=1, num_classes= num_classes, num_features=num_features, num_channels=num_channels)
    model.train(checkpoint_path='C:/tmp/'+datetime.now().strftime('%Y%m%d_%H%M%S'),epochs=1, log_path='C:/tmp/logs',train_data=train_data)
 
  
  

    
    
    
    
    

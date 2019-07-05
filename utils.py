# necessary python libraries
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Dense,LSTM,concatenate,Convolution1D,Dropout,Flatten,Reshape
from tensorflow.keras.models import Model


#%%
def get_pf(x_val,y_val,val_SNRs,model,epoch,pf_min,pf_max):
    '''
        callback for pfs evaluation at evert epoch end
    '''
    y_val_hat = model.predict(x_val,verbose=0)
    cm = confusion_matrix(np.argmax(y_val,1), np.argmax(y_val_hat, 1))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    pf = 100*cm_norm[1][0]
    print("False Alarm:%.3f%%"%pf)
    # set the pf stop interval for a CFAR detector
    if (pf>pf_min) & (pf<pf_max):
        print("Pf meet the threshold, training stopped")
        model.stop_training=True    

def performance_evaluation(save_path,x_test,y_test,test_SNRs,model):
    '''
        Evaluate final model's performance
    '''
    y_test_hat = model.predict(x_test,verbose=1)
    plt,pf=getConfusionMatrixPlot(np.argmax(y_test,1),np.argmax(y_test_hat,1))
    pd_list=[]
    snrs = np.linspace(-20,19,40)
    snrs = np.array(snrs,dtype='int16')
    for snr in snrs:
        test_x_i = x_test[np.where(test_SNRs==snr)]
        test_y_i = y_test[np.where(test_SNRs==snr)]
        test_y_i_hat = np.array(model.predict(test_x_i,verbose=0))
        cm = confusion_matrix(np.argmax(test_y_i, 1), np.argmax(test_y_i_hat,1))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        pd_list.append(cm_norm[0][0])     
    # save Pds result to xls file, the last element if Pf    
    import csv
    pd_list.append(pf)
    with open(save_path,'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(pd_list)
        
def getFontColor(value):
    '''
        set color in confusion matrix plot
    '''
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

def getConfusionMatrixPlot(true_labels, predicted_labels):
    '''
        plot confusion matrix
    '''
    import matplotlib.pyplot as plt
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)
    # add color bar
    plt.colorbar(res)
    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])
    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))
    # add genres as ticks
    alphabet = ['with signal','noise only'] 
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    plt.title('Confusion matrix for all snrs')
    print("Miss Detection:%.3f%%"%(100*cm_norm[0][1]))
    print("False Alarm:%.3f%%"%(100*cm_norm[1][0]))
    return plt,100*cm_norm[1][0]

def radioml_IQ_data(filename):
    '''
        load dataset for single node model training
    '''
    snrs=""
    mods=""
    lbl =""
    Xd = pickle.load(open(filename,'rb'),encoding='latin')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)
    
#     use QAM16 signal only
    lbl = np.array(lbl) 
    index = np.where(lbl=='QAM16')[0]   
    X = X[index]
    lbl = lbl[index]
    
    maxlen = X.shape[-1]
    SNR = []
    for item in lbl:
        SNR.append(item[-1])
    SNR = np.array(SNR,dtype='int16')
    
    noise_vectors = []
    for i in range(X.shape[0]):
        real = np.random.randn(maxlen) 
        imag = np.random.randn(maxlen)
        complex_noise_vector = real + 1j*imag
        energy = np.sum(np.abs(complex_noise_vector)**2)
        noise_vector = complex_noise_vector / (energy**0.5)
        real = np.real(noise_vector)
        imag = np.imag(noise_vector)
        noise_vectors.append([real,imag])
    noise_vectors = np.array(noise_vectors)   
    
    # one-hot label, [1,0] with signal, [0,1] noise only
    dataset = np.concatenate((X,noise_vectors),axis=0)
    labelset = np.concatenate(([[1,0]]*len(X),[[0,1]]*len(noise_vectors)),axis=0)
    labelset = np.array(labelset,dtype='int16')
    # use snr -100 to represent noise samples
    SNR = np.concatenate((SNR,[-100]*len(noise_vectors)),axis=0) 

    total_num = len(dataset)
    shuffle_idx = np.random.choice(range(0,total_num), size=total_num,replace=False)
    dataset = dataset[shuffle_idx]
    labelset = labelset[shuffle_idx]
    SNR = SNR[shuffle_idx]
    
    # split the whole dataset with ratio 3:1:1 into training, validation and testing set
    train_num = int(total_num*0.6)
    val_num = int(total_num*0.2)

    x_train = dataset[0:train_num]
    y_train = labelset[0:train_num]
    x_val = dataset[train_num:train_num+val_num]
    y_val = labelset[train_num:train_num+val_num]
    x_test = dataset[train_num+val_num:]
    y_test = labelset[train_num+val_num:]    
    val_SNRs = SNR[train_num:train_num+val_num]
    test_SNRs = SNR[train_num+val_num:]
    
    print("Training data:",x_train.shape)
    print("Training labels:",y_train.shape)
    print("Validation data:",x_val.shape)
    print("Validation labels:",y_val.shape)
    print("Testing data",x_test.shape)
    print("Testing labels",y_test.shape)

    return x_train,y_train,x_val,y_val,x_test,y_test,val_SNRs,test_SNRs 


def radioml_IQ_CO_data(filename):
    '''
        load dataset for cooperative model training
    '''
    snrs=""
    mods=""
    lbl =""
    Xd = pickle.load(open(filename,'rb'),encoding='latin')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)

#     use QAM16 signal only
    lbl = np.array(lbl) 
    index = np.where(lbl=='QAM16')[0]   
    X = X[index]
    lbl = lbl[index]
    
    maxlen = X.shape[-1]
    nodes = X.shape[1]
    
    SNR = []
    for item in lbl:
        SNR.append(item[-1])
    SNR = np.array(SNR,dtype='int16')
    
    noise_vectors = []
    for i in range(X.shape[0]*nodes):
        real = np.random.randn(maxlen) 
        imag = np.random.randn(maxlen)
        complex_noise_vector = real + 1j*imag
        energy = np.sum(np.abs(complex_noise_vector)**2)
        noise_vector = complex_noise_vector / (energy**0.5)
        real = np.real(noise_vector)
        imag = np.imag(noise_vector)
        noise_vectors.append([real,imag])
    noise_vectors = np.array(noise_vectors)   
    noise_vectors = np.reshape(noise_vectors,(int(noise_vectors.shape[0]/nodes),nodes,2,noise_vectors.shape[-1]))
    
    # one-hot label, [1,0] with signal, [0,1] noise only
    dataset = np.concatenate((X,noise_vectors),axis=0)
    labelset = np.concatenate(([[1,0]]*len(X),[[0,1]]*len(noise_vectors)),axis=0)
    labelset = np.array(labelset,dtype='int16')
    # use snr -100 to represent noise only samples
    SNR = np.concatenate((SNR,[-100]*len(noise_vectors)),axis=0) 
    
    print(dataset.shape)
    print(labelset.shape)
    print(SNR.shape)
    
    return dataset,labelset,SNR

def DetectNet(lr,input_shape,filter_num,lstm_units,kernel_size,drop_ratio,lstm_drop_ratio,dense_units):
    '''
        Network architecture of DetectNet
    '''
    ConvInput = Input(input_shape)
    x1 = Convolution1D(filter_num,kernel_size,padding='same',data_format='channels_first',activation='relu')(ConvInput)
    x1 = Dropout(rate=drop_ratio)(x1)
    x2 = Convolution1D(filter_num,kernel_size,padding='same',data_format='channels_first',activation='relu')(x1)
    x2 = Dropout(rate=drop_ratio)(x2)
    x3 = Flatten()(x2)
    # dimension reduction
    x4 = Dense(input_shape[-1],activation='linear')(x3)
    x4 = Reshape(target_shape=(1,input_shape[-1]))(x4)
    
    LSTMInput = concatenate([x4,ConvInput],axis=1)
    
    y1 = LSTM(units=lstm_units,return_sequences=True,recurrent_dropout=lstm_drop_ratio,input_shape=(input_shape[-1],3))(LSTMInput)
    y2 = LSTM(units=lstm_units,dropout=lstm_drop_ratio)(y1) 
    y2 = Flatten()(y2)
    y3 = Dense(dense_units,activation='relu')(y2)
    
    predictions = Dense(2, activation='softmax')(y3)
    model = Model(inputs=ConvInput,outputs=predictions)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model

def SoftCombinationNet(lr,input_shape,drop_ratio):
    '''
        Network architecture of SoftCombinationNet
    '''
    DenseInput = Input(input_shape)
    x1 = Flatten()(DenseInput)
    x1 = Dense(32,activation='relu')(x1)
    x1 = Dropout(rate=drop_ratio)(x1)
    x2 = Dense(8,activation='relu')(x1)
    x2 = Dropout(rate=drop_ratio)(x2)
    predictions = Dense(2, activation='softmax')(x2)
    model = Model(inputs=DenseInput,outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary() 
    return model

#%% other networks
def DNN(lr,input_shape,drop_ratio):
    DenseInput = Input(input_shape)
    x1 = Flatten()(DenseInput)
    x1 = Dense(256,activation='relu')(x1)
    x1 = Dropout(rate=drop_ratio)(x1)
    x2 = Dense(500,activation='relu')(x1)
    x2 = Dropout(rate=drop_ratio)(x2)
    x3 = Dense(250,activation='relu')(x2)
    x3 = Dropout(rate=drop_ratio)(x3)
    x4 = Dense(120,activation='relu')(x3)
    x4 = Dropout(rate=drop_ratio)(x4)   
    predictions = Dense(2, activation='softmax')(x4)
    model = Model(inputs=DenseInput,outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary() 
    return model

def CNN(lr,input_shape,filter_num,kernel_size,drop_ratio,dense_units): 
    ConvInput = Input(input_shape)
    x1 = Convolution1D(filter_num,kernel_size,padding='same',data_format='channels_first',
                                        input_shape=input_shape,activation='relu')(ConvInput)
    x1 = Dropout(rate=drop_ratio)(x1)
    x2 = Convolution1D(filter_num,kernel_size,padding='same',data_format='channels_first',
                                        activation='relu')(x1)
    x2 = Dropout(rate=drop_ratio)(x2)
    x2 = Flatten()(x2)

    x3 = Dense(dense_units,activation='relu')(x2)
    x3 = Dropout(rate=drop_ratio)(x3)
    predictions = Dense(2, activation='softmax')(x3)
    model = Model(inputs=ConvInput,outputs=predictions)
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model

def lstm(lr,input_shape):
    units = 128  #输出维度
    num_classes = 2
    lstm_inputs = Input(input_shape)
    x1 = LSTM(units=units,return_sequences=True,recurrent_dropout=0.2)(lstm_inputs)
    x2 = LSTM(units=units, dropout = 0.2)(x1) 
    predictions = Dense(num_classes, activation='softmax')(x2)
    model = Model(inputs=lstm_inputs,outputs=predictions)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.summary()
    return model

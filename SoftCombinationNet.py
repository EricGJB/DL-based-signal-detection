import utils
import numpy as np
import csv

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LambdaCallback,EarlyStopping,ModelCheckpoint

#%% single node sensing 
# hyperparameters
lr = 0.0003
drop_ratio = 0.2
sample_length = 128 
max_epoch = 100
batch_size = 200
patience = 8

# load data
dataset,labelset,SNR = utils.radioml_IQ_CO_data('pkl_data/'+str(sample_length)+'_co.pkl')
total_group = dataset.shape[0]
nodes = dataset.shape[1] 
total_num = total_group*nodes
   
snrs = np.linspace(-20,19,40)
snrs = np.array(snrs,dtype='int16')
snr_type = len(snrs)

# load single model
model_single = load_model('result/models/DetectNet/'+str(sample_length)+'/final.h5')
flatten_dataset = np.reshape(dataset,(total_num,2,sample_length))

predictions = model_single.predict(flatten_dataset,verbose=1)
decisions = np.argmax(predictions,axis=1)

noise_decisions = decisions[total_num//2:]
pf = 1 - np.mean(noise_decisions)

signal_decisions = np.reshape(decisions[:total_num//2],(snr_type,total_num//2//snr_type)) #按average snr来分
pd_list = np.zeros((snr_type,1))
i = 0
while i < snr_type:
    pd_list[i] = 1 - np.mean(signal_decisions[i])
    i = i + 1
    
pd_list = np.append(pd_list,pf)
with open('result/xls/SoftCombinationNet/Pds.xls','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(pd_list)
    
#%% cooperative sensing
noise_decisions_groups = np.reshape(noise_decisions,(noise_decisions.shape[0]//nodes,nodes))
# 1000 is the number of samples per specific modulation scheme and snr
signal_decisions_groups = np.reshape(signal_decisions,(snr_type,1000,nodes)) 

# Logical OR fusion rule
error = 0
for group in noise_decisions_groups:
    error = error + int(np.sum(group) < nodes)
pf_hard = error / (total_group//2)

pd_hard_list = np.zeros((snr_type,1))
i = 0
while i < snr_type:
    snr_decisions_groups = signal_decisions_groups[i]
    correct = 0
    for group in snr_decisions_groups:
        correct = correct + int(np.sum(group) < nodes)
    pd_hard_list[i] = correct / len(snr_decisions_groups)
    i = i + 1
    
pd_hard_list = np.append(pd_hard_list,pf_hard)
with open('result/xls/SoftCombinationNet/Pds_hard.xls','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(pd_hard_list)

# SoftCombinationNet
softmax_dataset = np.reshape(predictions,(total_group, nodes, 2))
shuffle_idx = np.random.choice(range(0,total_group), size=total_group,replace=False)
softmax_dataset = softmax_dataset[shuffle_idx]
SNR = SNR[shuffle_idx]
softmax_labelset = labelset[shuffle_idx]

co_x_train = softmax_dataset[:int(total_group*0.6)]
co_y_train = softmax_labelset[:int(total_group*0.6)]
co_x_val = softmax_dataset[int(total_group*0.6):int(total_group*0.8)]
co_y_val = softmax_labelset[int(total_group*0.6):int(total_group*0.8)]
co_x_test = softmax_dataset[int(total_group*0.8):]
co_y_test = softmax_labelset[int(total_group*0.8):]
val_SNRs = SNR[int(total_group*0.6):int(total_group*0.8)]
test_SNRs = SNR[int(total_group*0.8):]

input_shape = (nodes,2)
model_co = utils.SoftCombinationNet(lr,input_shape,drop_ratio)

early_stopping = EarlyStopping(monitor='val_loss',patience=patience)
best_model_path = 'result/models/SoftCombinationNet/best.h5'
checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True)
model_co.fit(co_x_train,co_y_train,epochs=max_epoch,batch_size=batch_size,verbose=1,shuffle=True,
             validation_data=(co_x_val, co_y_val),
             callbacks=[early_stopping,checkpointer])  

model_co = load_model(best_model_path)
pf_min = 1.5
pf_max = 2.5
pf_test = LambdaCallback(
    on_epoch_end=lambda epoch, 
    logs: utils.get_pf(co_x_val,co_y_val,val_SNRs,model_co,epoch,pf_min,pf_max))
model_co.fit(co_x_train,co_y_train,epochs=max_epoch,batch_size=batch_size,verbose=1,shuffle=True,
             callbacks=[pf_test])

utils.performance_evaluation('result/xls/SoftCombinationNet/Pds_soft.xls',co_x_test,co_y_test,test_SNRs,model_co)
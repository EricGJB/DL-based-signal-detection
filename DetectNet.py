import utils
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LambdaCallback,EarlyStopping,ModelCheckpoint,TensorBoard

# GPU usage setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%
# hyperparameters 
lr = 0.0003
filter_num = 60
kernel_size = 10
lstm_units = 128
drop_ratio = 0.2
lstm_drop_ratio = 0.2
dense_units = 128
sample_length = 128
max_epoch = 100
batch_size = 200
patience = 6

# load data
filename = 'pkl_data/'+str(sample_length)+'.pkl'
x_train,y_train,x_val,y_val,x_test,y_test,val_SNRs,test_SNRs = utils.radioml_IQ_data(filename)

# callbacks
early_stopping = EarlyStopping(monitor='val_loss',patience=patience)
best_model_path = 'result/models/DetectNet/'+str(sample_length)+'/best.h5'
checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True)
TB_dir = 'result/TB'
tensorboard = TensorBoard(TB_dir)
model = utils.DetectNet(lr,(2,sample_length),filter_num,lstm_units,kernel_size,drop_ratio,lstm_drop_ratio,dense_units)
history = model.fit(x_train,y_train,epochs=max_epoch,batch_size=batch_size,verbose=1,shuffle=True,validation_data=(x_val, y_val),
                    callbacks=[early_stopping,checkpointer,tensorboard])  
print('Fisrt stage finished, loss is stable')


pf_min = 6.5
pf_max = 7.5
pf_test = LambdaCallback(
    on_epoch_end=lambda epoch, 
    logs: utils.get_pf(x_val,y_val,val_SNRs,model,epoch,pf_min,pf_max))
print('Start second stage, trade-off metrics')
model = load_model(best_model_path)
model.fit(x_train,y_train,epochs=max_epoch,batch_size=200,verbose=1,shuffle=True,
          callbacks=[pf_test])
if model.stop_training:
    # save results
    model.save('result/models/DetectNet/'+str(sample_length)+'/final.h5')
    print('Second stage finished, get the final model')
    save_path = 'result/xls/DetectNet/'+str(sample_length)+'/Pds.xls'
    utils.performance_evaluation(save_path,x_test,y_test,test_SNRs,model)
else:
    print("Can't meet pf lower bound")

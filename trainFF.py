import numpy as np
import pandas as pd
import os
import time
#os.chdir('/Users/mr54725/Dropbox/Research/BellLabs_pathloss/terrain/')
from modelsNewEncode import *
import pickle


from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold

# Importing gc module
import gc

# Declare constants
c=3e8
freq =1.8e9
lam=c/freq
GuillaumeSite=[611981,3168407]
BSloc=GuillaumeSite


desc = 'dense_9'

# Load data
columns = ['ht', 'loc_x','loc_y', 'measPathGain', 'profile','ranges','terrainHtFile','curv', 'terrainDev', 'predPathGain', 'dist']
data = pd.DataFrame(np.load('data/all_dataset.npy'), columns = columns)
numericas = ['ht', 'loc_x','loc_y', 'measPathGain','terrainHtFile','curv', 'terrainDev', 'predPathGain', 'dist']
data = data.astype({col: np.float32 for col in numericas})
ceil = 2*10*np.log10(lam/(4*np.pi*np.array(data.dist, dtype = np.float32)))

X = pad_sequences(data.profile)
X = np.append(X, np.expand_dims(data.dist.values, axis = 1), axis = 1)
y = np.array(data.measPathGain)


if 'conv' in desc:
  X = np.expand_dims(X, axis=2)

kfold = KFold(n_splits=5, shuffle = True,  random_state=42)

cvscores = []
fold = 1

print(' \n Started training \n ')
print(' \n Model:    ' + desc)
print('\n')

t0 = time.time()

predictions = []
for train, test in kfold.split(X, y):
  tf1 = time.time()
# create model
  desc_f = desc + str(fold) + '_' 
  model = dense_9()

  filepath="modelsNN/newEncode_" + desc_f + time.strftime('%l_%M%p_%b_%d_%Y')[1:] + "-{epoch:02d}-{val_loss:.2f}.hdf5"
  callback = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)
  


  history = model.fit(X[train], y[train], 
                    validation_data = (X[test], y[test]), 
                    epochs=300, batch_size=32, 
                    callbacks = [callback],
                    verbose=0)

  pickle.dump(history.history, open('resultsCV/NEWhistory_'+desc_f+ '.p', 'wb'))
  pred = model.predict(X[test])
  predictions.append(pred)
  scores = model.evaluate(X[test], y[test], verbose=0)
  cvscores.append(scores)
  tf2 = time.time()
  print('finished fold:  ' + str(fold) + '  -  Time:  ' + str(tf2-tf1) + '  -  score:  ' + str(scores) )
  fold = fold+1
  gc.collect()





pickle.dump(predictions, open('resultsCV/predictions_'+desc+ '.p', 'wb'))
tf = time.time()
time_file = np.array([t0,tf,tf-t0])
np.save('resultsCV/' +desc+'_cvresults.npy',cvscores)

np.save('resultsCV/' +desc+'_time.npy',time_file)


print(' \n CV mean rmse:  '+ str(np.mean(np.sqrt(cvscores)))+ '  +/-  ' +str(np.std(np.sqrt(cvscores))) + '\n')


print('  \n Mean time:  ' + str((tf-t0)/5) + '  \n ' )


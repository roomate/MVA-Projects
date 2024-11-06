# Import librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Download data
Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()

# Normalize it
min_Xtr = np.min(Xtr, axis=0)
max_Xtr = np.max(Xtr, axis=0)
Xtr = (Xtr - min_Xtr)/(max_Xtr - min_Xtr)

min_Xte = np.min(Xte, axis=0)
max_Xte = np.max(Xte, axis=0)
Xte = (Xte - min_Xte)/(max_Xte - min_Xte)

# Ensure the data are balanced
for i in range(10):
  print("nombre de", i , ":" ,np.count_nonzero(Ytr == i))
# Analyse the data
print("dimension of X_train: ", Xtr.shape)
print("first image: " , Xtr[0])
print("maximum value: ", max(Xtr[0]), "minimum val)

# Evaluate the accuracy of the model on validation data
def acc(pred, label):
  N = len(label)
  res = pred - label
  res = np.count_nonzero(res==0)
  return res / N


#############
#### Transform the data in a valid image format

#### training data
red_channel_tr = Xtr[:, :1024]
green_channel_tr = Xtr[:, 1024:2048]
blue_channel_tr = Xtr[:, 2048:]
red_channel_matrix_tr = red_channel_tr.reshape(5000,32, 32)
green_channel_matrix_tr = green_channel_tr.reshape(5000, 32, 32)
blue_channel_matrix_tr = blue_channel_tr.reshape(5000, 32, 32)

#### test data
red_channel_test = Xte[:, :1024]
green_channel_test = Xte[:, 1024:2048]
blue_channel_test = Xte[:, 2048:]
red_channel_matrix_test = red_channel_test.reshape(2000,32, 32)
green_channel_matrix_test = green_channel_test.reshape(2000, 32, 32)
blue_channel_matrix_test = blue_channel_test.reshape(2000, 32, 32)

# Combine each color and create Gray scale image (sum of each color intensities)
images = np.stack((red_channel_matrix_tr, green_channel_matrix_tr, blue_channel_matrix_tr),axis=-1)
images = np.mean(images,-1)
images_test = np.stack((red_channel_matrix_test, green_channel_matrix_test, blue_channel_matrix_test),axis=-1)
images_test = np.mean(images_test,-1)

#############

#### Extract the features: HOG + hist of each row  -----> Training data
Xtr_HOG = []
for j in range(5000):
   Xtr_HOG.append(hog_features(images[j], 9, (8,8), (1,1)))
Xtr_HOG = np.array(Xtr_HOG)

Xtr_hist = []
for j in range(5000):
  Xtr_hist.append(Hist_im(images[j] , n_bins=12))
Xtr_hist = np.array(Xtr_hist)


#### Extract the features: HOG + hist of each row  -----> Testing data
Xtest_HOG = []
for j in range(2000):
   Xtest_HOG.append(hog_features(images_test[j], 9, (8,8), (1,1)))
Xtest_HOG = np.array(Xtest_HOG)

Xtest_hist = []
for j in range(2000):
   Xtest_hist.append(Hist_im(images_test[j],n_bins=12))
Xtest_hist = np.array(Xtest_hist)

#### Prepare the data
Xtr_HOG_H = np.concatenate([Xtr_HOG, Xtr_hist],axis=1)
Xtest_HOG_H = np.concatenate([Xtest_HOG, Xtest_hist],axis=1)

# Divide the data in a validation set and in a training set
indices = np.arange(3000)
np.random.shuffle(indices)
X_shuffled = Xtr_HOG_H.iloc[indices]
y_shuffled = Ytr.iloc[indices]

test_size = int(0.5 * 3000)

xtr = X_shuffled[test_size:]
ytr = y_shuffled[test_size:]
xval = X_shuffled[:test_size]
yval = y_shuffled[:test_size]

# Perform a first experiment on our data to see the accuracy on the validation data
svm.train(xtr, ytr)
pred = model.predict(xval)
print("accuracy: ", acc(pred,yval))

#### Compute multi_class SVM and save the result as dataframe
svm = SVM(gauss_kernel , epsilon = 1e-13 , C=1)
svm.train(xtr, ytr)
Yte = svm.predict(xval)
Yte = {'Prediction' : Yte}
dataframe = pd.DataFrame(Yte)
dataframe = dataframe.astype(int)
dataframe.index += 1
dataframe.to_csv('Yte_pred.csv',index_label='Id')


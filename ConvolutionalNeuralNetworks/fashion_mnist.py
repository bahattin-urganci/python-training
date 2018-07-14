from keras.datasets import fashion_mnist

(train_x,train_y),(test_x,test_y)=fashion_mnist.load_data()

import numpy as np 
from keras.utils import to_categorical
import matplotlib.pyplot as plt


print('Training data shape : ', train_x.shape, train_y.shape)

print('Testing data shape : ', test_y.shape, test_y.shape)
num_classes=10


#train labellarından eşsiz olanları bulacaz
classes=np.unique(train_y)
nClasses=len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[5,5])

#ilk imaj neymiş bakalım bi
plt.subplot(121)
plt.imshow(train_x[0,:,:],cmap='gray')
plt.title("Ground truth : {}".format(train_y[0]))


# test datasındaki ilk imaja bakıyoz
plt.subplot(122)
plt.imshow(test_x[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_y[0]))

plt.show()
#data preprocessing
#datayı 28 x 28 x 1 şeklinde matrise dönüştürüyoruz 
train_x=train_x.reshape(-1,28,28,1)
test_x=test_x.reshape(-1,28,28,1)
train_x=train_x.astype('float32')
test_x=test_x.astype('float32')
train_x=train_x/255
test_x=test_x/255

#şimdi kategorize edilmişleri one-hot-encoding tekniği ile vectörlere çevirecez

train_y_one_hot=to_categorical(train_y)
test_y_one_hot=to_categorical(test_y)

# Değişiklik öncesi ve sonrasını gösterir
print('Original label:', train_y[0])
print('After conversion to one-hot:', train_y_one_hot[0])

from sklearn.model_selection import train_test_split
train_x,valid_x,train_label,valid_label=train_test_split(train_x,train_y_one_hot,test_size=0.2,random_state=13)
#data preprocessing bitti


#network


#modeling the data
import keras

from keras.models import Sequential,Input,Model,load_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

#
#batch_size = 64
#epochs = 20
#num_classes = 10
#
#fashion_model = Sequential()
#fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
#fashion_model.add(LeakyReLU(alpha=0.1))
#fashion_model.add(MaxPooling2D((2, 2),padding='same'))
#fashion_model.add(Dropout(0.25))
#fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
#fashion_model.add(LeakyReLU(alpha=0.1))
#fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#fashion_model.add(Dropout(0.25))
#fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
#fashion_model.add(LeakyReLU(alpha=0.1))                  
#fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#fashion_model.add(Dropout(0.4))
#fashion_model.add(Flatten())
#fashion_model.add(Dense(128, activation='linear'))
#fashion_model.add(LeakyReLU(alpha=0.1))           
#fashion_model.add(Dropout(0.3))
#fashion_model.add(Dense(num_classes, activation='softmax'))
#fashion_model.summary()
#
#fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#fashion_train_dropout = fashion_model.fit(train_x, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x, valid_label))
#
#   fashion_model.save("/ConvolutionalNeuralNetworks/fashion_model_dropout.h5py")
#
#test_eval = fashion_model.evaluate(test_x, test_y_one_hot, verbose=1)
#
#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])
#
#accuracy = fashion_train_dropout.history['acc']
#val_accuracy = fashion_train_dropout.history['val_acc']
#loss = fashion_train_dropout.history['loss']
#val_loss = fashion_train_dropout.history['val_loss']
#epochs = range(len(accuracy))
#plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
#plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy')
#plt.legend()
#plt.figure()
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()


fashion_model = load_model('ConvolutionalNeuralNetworks/fashion_model_dropout.h5py')

test_eval=fashion_model.evaluate(test_x,test_y_one_hot,verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


#accuracy = fashion_model.history['acc']
#val_accuracy = fashion_model.history['val_acc']
#loss = fashion_model.history['loss']
#val_loss = fashion_model.history['val_loss']
#epochs = range(len(accuracy))
#plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
#plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy')
#plt.legend()
#plt.figure()
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()


# predicting labels
predicted_classes=fashion_model.predict(test_x)

predicted_classes=np.argmax(np.round(predicted_classes),axis=1)

correct=np.where(predicted_classes==test_y)[0]
print("Found %d correct labels" % len(correct))

#for i,correct in enumerate(correct[:9]):
#    plt.subplot(3,3,i+1)
#    plt.imshow(test_x[correct].reshape(28,28), cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct]))
#    plt.tight_layout()
#
#
#plt.show()
incorrect=np.where(predicted_classes!=test_y)[0]
print("Found %d incorrect labels" % len(incorrect))

#for i,incorrect in enumerate(incorrect[:9]):
#    plt.subplot(3,3,i+1)
#    plt.imshow(test_x[incorrect].reshape(28,28), cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_y[incorrect]))
#    plt.tight_layout()
#
#plt.show()

#sınıflandırma raporu
from sklearn.metrics import classification_report
target_names=["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_y,predicted_classes,target_names=target_names))

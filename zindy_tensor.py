import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Test dataset

df_test = pd.read_csv('.\zindi\Test.csv')

df_test['Target'] = df['Target'].astype('int')
X = df_test.drop('User_ID', axis=1)
X_t = X.values
df_test['Target'] = df['Target'].astype('int')
y_t = df_test['Target'].values
df_test = df_test.values
X = df_test[:,1:-1]
y = df_test[:,-1]
# y = df_test['Target'].values
print(X, y.T)
print(type(X), type(y))
# Training dataset
useful = ['month','year','CompPart','Comment','Sub','Disc','Target']
df = pd.read_csv('.\zindi\Train.csv')
X = df[useful].values
# X = df.drop('Target', axis=1)
# X_train = features.values
# Xtr_train = X.values
y = df['Target']
y = y.values
# y_train = target.values


# To split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=1)
# Build TF NN, batch_size = 30, epoch=100

nn = tf.keras.models.Sequential()
input = X_train.shape[1]

nn.add(Dense(1024, input_shape = (input,), activation ='relu'))
nn.add(Dropout(0.2))
nn.add(Dense(1024, activation = 'relu'))
nn.add(Dropout(0.2))
nn.add(Dense(1, activation ='sigmoid'))
nn.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
nn.summary()

mc = ModelCheckpoint('model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 0, save_best_only = True)

# to split dataset

to_fit_nn = nn.fit(X_train, y_train, epochs =10, 
                   validation_data = (X_train, y_train), batch_size = 100, callbacks = [mc])

best = load_model('model.h5')
accuracy = best.evaluate(X_train, y_train, verbose=False)
print("Best model training score: {:.4f}".format(accuracy[0]))
print("Best model training score: {:.4f}".format(accuracy[1]))

accuracy = best.evaluate(X_test, y_test, verbose=False)
print("Best model training score: {:.4f}".format(accuracy[0]))
print("Best model training score: {:.4f}".format(accuracy[1]))
print("------------------------------------------------------>")

accuracy = nn.evaluate(X_train, y_train, verbose=False)
print("Final model training score: {:.4f}".format(accuracy[0]))
print("Final model training score: {:.4f}".format(accuracy[1]))

accuracy = nn.evaluate(X_test, y_test, verbose=False)
print("Final model training score: {:.4f}".format(accuracy[0]))
print("Final model training score: {:.4f}".format(accuracy[1]))
def history_of_plot(fitting):
    bin = fitting.history['accuracy']
    valbin = fitting.evaluate['val_accuracy']
    lossy = fitting.history['loss']
    vallossy = fitting.evaluate['val_loss']
    x = range(1, len(bin) + 1)
    
    plt.figure(figsize=(14,7))
    plt.subplot(1,2,1)
    plt.plot(x, bin, 'b', label='Training Accuracy')
    plt.plot(x, valbin, 'r', label='Testing Accuracy')
    plt.title('Training and testing Accuracy')
    plt.legend()
    
    plt.figure(figsize=(14,7))
    plt.subplot(1,2,2)
    plt.plot(x, lossy, 'b', label='Training loss')
    plt.plot(x, vallossy, 'r', label='Testing loss')
    plt.title('Training and testing Loss')
    plt.legend()
    
history_of_plot(to_fit_nn)

:


history_of_plot(to_fit_nn)



In [ ]:


# Making predictions based on classes and probabilities
y_prednn = best.predict(X_test)
y_prednn_prob = best.predict_proba(X_test)
y_classes_prednn = best.predict_classes(X_test)
​
#1d array
y_prednn_1 = y_prednn[:,0]
y_classes_prednn_1 = y_classes_prednn[:,0]

# Making predictions based on classes and probabilities
y_prednn = best.predict(X_test)
y_prednn_prob = best.predict_proba(X_test)
y_classes_prednn = best.predict_classes(X_test)

#1d array
y_prednn_1 = y_prednn[:,0]
y_classes_prednn_1 = y_classes_prednn[:,0]

# Print Network evaluation metrics
nn_confuse = confusion_matrix(y_test, y_classes_prednn_1)
print(f'Confusion matrix for our network:\n{nn_confuse}')
print(---------------------------------------------------)

nn_accuracy = accuracy_score(y_test, y_classes_prednn_1)
print('The Accuracy score: %f' % nn_accuracy)
print(---------------------------------------------------)

auc_nn = roc_auc_score(y_test, y_classes_prednn_1)
print('ROC AUC: %f' %roc_auc_nn)
print(---------------------------------------------------)

precision_nn = precision_score(y_test, y_classes_prednn_1)
print('Precision %f' %precision_nn)
print(---------------------------------------------------)

f1_nn = f1_score(y_test, y_classes_prednn_1)
print('F1 score result %f' %f1_nn)
print(---------------------------------------------------)

recall_nn = recall_score(y_test, y_classes_prednn_1)
print('The Recall score: %f' % recall_nn)

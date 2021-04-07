import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 


#for reproductibility 
np.random.seed(42)

train_df=pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\datascience\\dataset\\train.csv')

y=train_df['label']
x=train_df.drop(labels=['label'], axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x ,y,test_size = 0.4, random_state = 10)

#normalize
x= x / 255.0
x_test = x_test / 255.0
#convert to n-dimensional array
x = x.values
x_test = x_test.values

# visualize
plt.figure
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x[i].reshape(28, 28), cmap='gray')
    plt.title(y[i])
    plt.axis('off')
    plt.savefig('mnist_plot.png')
plt.show()

#training a KNN Classifier with Default K Value
knn_clf=KNeighborsClassifier()
knn_clf.fit(x_train,y_train)

preds = knn_clf.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)

acc_list = []
#testing knn classifier with 20 k values
for i in range(1, 21):
    knn_clf = KNeighborsClassifier(n_neighbors=i)
    
    knn_clf.fit(x, y)
    
    preds = knn_clf.predict(x_test)
    
    accuracy = accuracy_score(y_test, preds)
    
    acc_list.append(accuracy)


#plotting accuracy vs number of neighbors
num_k = np.arange(1, 21)
plt.figure(dpi=100)
plt.style.use('ggplot')
plt.plot(num_k, acc_list)
plt.xlabel('Number of Neihgbors')
plt.ylabel('Accuracy %')
plt.savefig('acc_plot.png')
plt.show() 
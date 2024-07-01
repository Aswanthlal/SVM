import numpy as np
import pandas as pd
import pylab as pl
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#SVM
#SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, 
#even when the data are not otherwise linearly separable. A separator between the categories is found, 
#then the data is transformed in such a way that the separator could be drawn as a hyperplane. 
#Following this, characteristics of new data can be used to predict the group to which a new record should belong.


#lodaing dataset
cell_df=pd.read_csv('cell_samples.csv')
cell_df.head()
#The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics.
#The ID field contains the patient identifiers. The characteristics of the cell samples from each patient are contained in fields Clump to Mit. 
#The values are graded from 1 to 10, with 1 being the closest to benign.
#The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).


#distribution based on clump thickness and uniformity of cell
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()


#data preprocessing and selection

#BareNuc column includes some values that are not numerical
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

#We want the model to predict the value of Class (that is, benign (=2) or malignant (=4)).
y=np.asanyarray(cell_df['Class'])
y[0:5]

#train/test dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Modeling (svm with scikit-learn)

#use the default, RBF (Radial Basis Function)
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

#predict
yhat = clf.predict(X_test)
yhat [0:5]


#Evaluation
from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

#f1 score
from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 

#jaccard index
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=2)


#rebuild the model with a linear kernel 
clf2 = svm.SVC(kernel='linear')
clf2.fit(x_train, y_train) 
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))

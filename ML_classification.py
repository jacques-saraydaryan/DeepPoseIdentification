# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import *
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d


def load_data(pickle_file):
    #load pickle file and convert into np array
    print('Loading data...')
    dataset = pd.read_pickle(pickle_file)

    X = np.array(dataset)[:,:-2]
    y = np.array(dataset)[:,-1] 

    #split data between train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    return X_train, X_test, y_train, y_test, dataset 

def plot_ROC(X_test,y_test,clf):
    y_score = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test,y_score[:,1])
    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()



def classificatio_test(X_train, X_test, y_train, y_test):
    PLOT_ROC=False

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    # iterate over classifiers and print results
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('Accuracy for',name,'is: ',score)
        if PLOT_ROC :
            plot_ROC(X_test,y_test,clf)
        #y_pred = clf.predict(X_test)
        #conf_matrix = confusion_matrix(y_test,y_pred)

    
def tsne_proj(dataset):
    #TSNE projection 
    print('TSNE projection...')
    #select data according to label
    data = np.array(dataset)[:-1]
    X_0 = data[data[:,-1]==0]
    X_1 = data[data[:,-1]==1]


    X_embedded0 = TSNE(n_components=3).fit_transform(X_0)
    X_embedded1 = TSNE(n_components=3).fit_transform(X_1)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot3D(X_embedded0[:,0],X_embedded0[:,1],X_embedded0[:,2],'.',label='Focus')
    ax.plot3D(X_embedded1[:,0],X_embedded1[:,1],X_embedded1[:,2],'.',label='Distract')
    #plt.plot(X_embedded0[:,0],X_embedded0[:,1],'.',label='Focus')
    #plt.plot(X_embedded1[:,0],X_embedded1[:,1],'.',label='Distract')
    #plt.legend()
    plt.show()

 
if __name__ == '__main__':

    X_train, X_test, y_train, y_test, dataset = load_data('data.pkl')
    classificatio_test(X_train, X_test, y_train, y_test)
    tsne_proj(dataset)

from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from hog import HOG
from data_loader import Dataset
import argparse
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import itertools
import numpy as np

#==========================================================================================================================================
#FILL THE FOLLOWING VARIABLES WITH YOUR DIRECTORY/INFO

myDirectory = '[FILL THIS IN]'



def plot_confusion_matrix(cm, classes=['inflamed aorta', 'negative'],
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


ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", default="svm.pickle",
                help="path to where the model will be stored")
args = vars(ap.parse_args())
print("Collecting annotations ...")


#CHANGE 'inflammed aorta' to the disease which you are working to diagnose
d = Dataset(myDirectory,
            myDirectory, ['inflamed aorta'])
labels, images = d.load_data()
print("Gathered {} image slices".format(len(images)))
data = []
labels_new = []

hog = HOG(orientations=19, pixelsPerCell=(8, 8),
          cellsPerBlock=(3, 3), transform=True)

for i, image in enumerate(images):
    if i % 100 == 0:
        print("Gathering features, {} of {}".format(i, len(images)))
    if 0 not in image.shape:
        image_resized = resize(image, (291, 218), anti_aliasing=True)
        hist = hog.describe(rgb2gray(image_resized))
        data.append(hist)
        labels_new.append(labels[i])

X_train, X_test, y_train, y_test = train_test_split(data, labels_new, random_state=0)
print("Training on {} images".format(len(X_train)))
print("Testing on  {} images".format(len(X_test)))

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      title='Confusion matrix, without normalization')
plt.show()
print("Accuracy Score: {:.2f}".format(accuracy_score(y_test, y_pred)))

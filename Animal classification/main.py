import matplotlib.pyplot as plt
import numpy as np
import os

import joblib
from skimage.io import imread
from skimage.transform import resize

def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
    print("Resized image pickle file not found... Resizing images.")
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:-4])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)

    print("Finished resizing images.")
 
data_path = './data/'

base_name = 'animal_faces'
# width = 160
width = 80

include = set(os.listdir(data_path))

try:
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    print("Resized image pickle file found.")
except FileNotFoundError:
    resize_all(src=data_path, pklname=base_name, width=width, include=include)
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')

from collections import Counter
 
Counter(data['label'])

X = np.array(data['data'])
y = np.array(data['label'])

from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)

from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
 
# create an instance of each transformer
grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(14, 14), 
    cells_per_block=(2,2), 
    orientations=9, 
    block_norm='L2-Hys'
)
scalify = StandardScaler()
 
# call fit_transform on each transform converting X_train step by step
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_prepared, y_train)

X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def optimize_HOG():
    print("HOG pickle file not found... Optimizing parameters.")
    HOG_pipeline = Pipeline([
        ('grayify', RGB2GrayTransformer()),
        ('hogify', HogTransformer(
            pixels_per_cell=(14, 14), 
            cells_per_block=(2, 2), 
            orientations=9, 
            block_norm='L2-Hys')
        ),
        ('scalify', StandardScaler()),
        ('classify', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
    ])
    
    clf = HOG_pipeline.fit(X_train, y_train)
    print('Percentage correct: ', 100*np.sum(clf.predict(X_test) == y_test)/len(y_test))
    
    param_grid = [
        {
            'hogify__orientations': [8, 9],
            'hogify__cells_per_block': [(2, 2), (3, 3)],
            'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)]
        },
        {
            'hogify__orientations': [8],
            'hogify__cells_per_block': [(3, 3)],
            'hogify__pixels_per_cell': [(8, 8)],
            'classify': [
                SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
                svm.SVC(kernel='linear')
            ]
        }
    ]

    grid_search = GridSearchCV(HOG_pipeline, 
                            param_grid, 
                            cv=3,
                            n_jobs=-1,
                            scoring='accuracy',
                            verbose=1,
                            return_train_score=True)
    
    grid_res = grid_search.fit(X_train, y_train)

    joblib.dump(grid_res, f'hog_sgd_model_{width}x{width}px.pkl')
    print("Finished Optimizing HOG parameters.")

try:
    grid_res = joblib.load(f'hog_sgd_model_{width}x{width}px.pkl')
    print("HOG pickle file found.")
except FileNotFoundError:
    optimize_HOG()
    grid_res = joblib.load(f'hog_sgd_model_{width}x{width}px.pkl')

best_pred = grid_res.predict(X_test)
print('Estimated percentage correct: ', 100*np.sum(best_pred == y_test)/len(y_test))

# importing cv2 
import cv2 

window_name = "HOG test"

# Create images from test data
for i in range(len(X_test)):
    image = cv2.resize(X_test[i], (512, 512))
    
    # Add the predicted label to the image
    cv2.putText(image, "Prediction: " + str(best_pred[i]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Add the real label to the image
    cv2.putText(image, "Real: " + str(y_test[i]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow(window_name, image)

    # Wait for keystroke
    cv2.waitKey(0)

    # Close window
    cv2.destroyWindow(window_name)
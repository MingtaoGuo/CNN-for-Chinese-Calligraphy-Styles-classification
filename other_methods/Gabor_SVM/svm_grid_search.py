import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import datetime as dt


from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata

# ---------------- classification begins -----------------
# scale data for [0,255] -> [0,1]
# sample smaller size for testing
# rand_idx = np.random.choice(images.shape[0],10000)
# X_data =images[rand_idx]/255.0
# Y      = targets[rand_idx]

# full dataset classification

data = sio.loadmat("gabor.mat")
traindata = data["traindata"]
trainlabel = np.squeeze(data["trainlabel"], axis=0)
testdata = data["testdata"]
testlabel = np.squeeze(data["testlabel"], axis=0)

# X_data = images / 255.0
# Y = targets

# split data to train and test
# from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)

############### Classification with grid search ##############

# Warning! It takes really long time to compute this about 2 days

# Create parameters grid for RBF kernel, we have to set C and gamma
from sklearn.model_selection import GridSearchCV

# generate matrix with gammas
# [ [10^-4, 2*10^-4, 5*10^-4],
#   [10^-3, 2*10^-3, 5*10^-3],
#   ......
#   [10^3, 2*10^3, 5*10^3] ]
# gamma_range = np.outer(np.logspace(-4, 3, 8),np.array([1,2, 5]))


# generate a much smaller matrix with gammas
# it is essentially the same operation as above
# but with the smaller range of parameters
# it will be faster
# [ [10^-3,  5*10^-3],
#   [10^-2,  5*10^-2],
#   ......
#   [10^0, 5*10^0] ]
gamma_range = np.outer(np.logspace(-3, 0, 4), np.array([1, 5]))

# make matrix flat, change to 1D numpy array
gamma_range = gamma_range.flatten()
gamma_range = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])

# generate matrix with all C
# C_range = np.outer(np.logspace(-3, 3, 7),np.array([1,2, 5]))
C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 5]))
# flatten matrix, change to 1D numpy array
C_range = C_range.flatten()
C_range = np.array([1e-1, 1, 10, 100])

parameters = {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range}

svm_clsf = svm.SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=1, verbose=2)

start_time = dt.datetime.now()
print('Start param searching at {}'.format(str(start_time)))

grid_clsf.fit(traindata, trainlabel)

elapsed_time = dt.datetime.now() - start_time
print('Elapsed time, param searching {}'.format(str(elapsed_time)))
sorted(grid_clsf.cv_results_.keys())

classifier = grid_clsf.best_estimator_
params = grid_clsf.best_params_

scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                          len(gamma_range))

# plot_param_space_heatmap(scores, C_range, gamma_range)

######################### end grid section #############

# Now predict the value of the test
expected = testlabel
predicted = classifier.predict(testdata)
print("Test Accuracy: %4g"%(np.mean(np.int32(predicted == expected))))
# show_some_digits(X_test, predicted, title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

# plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
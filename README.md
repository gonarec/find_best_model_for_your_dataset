# WineDetector_Python_Project

This Python project is built to classify a numeric dataset. 
It's possibile to remove the Nan value with different methods.
It's possibile to remove the outliners value with different methods.
It's possibile to setup the algo with different parameters.
The classification is obtain through four different algo: 
SVM, Neural Network, Bayes Gaussian and Decision Tree from scikit.learn library (sklearn).
Results from classification are rappresented with 4 different metrics:
accuracy, recall, precision and F1score.

class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)

    Cfloat, default=1.0    
    The regularization parameter. A larger C aims to correctly classify all training points by reducing the separation margin, which can lead to a higher risk of overfitting. Conversely, a smaller C increases the separation margin but allows for some classification errors, which can improve the model's generalization abilit

    kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’

class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

    hidden_layer_sizes: Specifies the number and size of the hidden layers in the neural network. The greater the number of neurons and layers, the higher the network's ability to learn complex representations, but this can also increase the risk of overfitting

    The activation function for the hidden layers. Common choices are:

        'identity': No activation function, f(x) = x.
        'logistic': Sigmoid function, f(x) = 1 / (1 + exp(-x)).
        'tanh': Hyperbolic tangent function, f(x) = tanh(x).
        'relu': Rectified Linear Activation function, f(x) = max(0, x).

class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, monotonic_cst=None)

    criterion{“gini”, “entropy”, “log_loss”}, default=”gini”

    splitter{“best”, “random”}, default=”best”
    The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.

class sklearn.naive_bayes.GaussianNB(*, priors=None, var_smoothing=1e-09)

    


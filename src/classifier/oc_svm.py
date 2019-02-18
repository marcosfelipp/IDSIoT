from sklearn import svm


def one_class(device_train, device_test, outlaier_test):
    """
    Train SVM with device set and test
    :param device_train: vector with features of device train set
    :param device_test: vector with features of device test set
    :param outlaier_test: vector with features of outlier device
    :return: Qtd. of device errors and Qtd. of outlier errors
    """

    clf = svm.OneClassSVM(nu=0.055, kernel="rbf", gamma=5)

    clf.fit(device_train)

    pred_test = clf.predict(device_test)
    pred_outliers = clf.predict(outlaier_test)

    n_error_test = pred_test[pred_test == -1].size
    n_error_outliers = pred_outliers[pred_outliers == 1].size

    return n_error_test, n_error_outliers

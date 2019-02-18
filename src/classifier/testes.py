from src.classifier.oc_svm import one_class
from src.classifier.data_manipulation import DataManipulation


def test_svm():

    vetor_features_train, classes = DataManipulation.prepare_dataset_features('features_novas/features_normal')
    vetor_features_outlier, classes = DataManipulation.prepare_dataset_features('features_novas/features_attack')
    vetor_features_test = []

    # TODO : Implementar aqui o particionamento (trainamento/teste)

    for i in range(200):
        vetor_features_test.append(vetor_features_train[0])
        del vetor_features_train[0]

    error_device, error_outlier = one_class(vetor_features_train, vetor_features_test, vetor_features_outlier)

    print(vetor_features_test[1])

    print("Error device: " + str(error_device/len(vetor_features_test)))
    print("Error outlier: " + str(error_outlier / len(vetor_features_outlier)))


if __name__ == "__main__":
    test_svm()

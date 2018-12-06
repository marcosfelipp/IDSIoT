import random

features_normal = './features_novas/' + 'features_normal' + '.arff'
features_attack = './features_novas/' + 'features_attack' + '.arff'

features_train = './features_novas/' + 'features_train' + '.arff'
features_test = './features_novas/' + 'features_test' + '.arff'

features_test_all = './features_novas/' + 'features_test_all' + '.arff'


def create_train_test():
    vector_features_attack = []
    vector_features_normal = []

    vector_features_train = []
    vector_features_test = []

    qtd_normal_train = 600
    qtd_attack_train = 100

    with open(features_attack) as attack, open(features_normal) as normal,  open(features_train, 'w') as train, open(features_test, 'w') as test:
        for line in attack:
            vector_features_attack.append(line)

        for line in normal:
            vector_features_normal.append(line)

        #     Verify repited tuples:
        _normal = []
        _attack = []
        for line in vector_features_normal:
            if line in _normal:
                pass
            else:
                _normal.append(line)

        for line in vector_features_attack:
            if line in _attack:
                pass
            else:
                _attack.append(line)

        for i in range(qtd_attack_train):
            sorted = random.randint(0, len(_attack)-1)
            vector_features_train.append(_attack.pop(sorted))

        for i in range(qtd_normal_train):
            sorted = random.randint(0, len(_normal)-1)
            vector_features_train.append(_normal.pop(sorted))

        for i in range(len(_attack)):
            vector_features_test.append(_attack[i])

        for i in range(len(_normal)):
            vector_features_test.append(_normal[i])

        random.shuffle(vector_features_train)
        # random.shuffle(vector_features_test)

        train.writelines(vector_features_train)
        test.writelines(vector_features_test)


def create_all_dataset():
    with open(features_attack) as attack, open(features_normal) as normal,  open(features_test_all, 'w') as test:
        vector_features_all = []

        for line in attack:
            vector_features_all.append(line)

        for line in normal:
            vector_features_all.append(line)

        test.writelines(vector_features_all)

if __name__ == "__main__":
    create_all_dataset()

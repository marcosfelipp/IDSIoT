from random import randint

features_normal = './features_novas/' + 'features_normal' + '.arff'
features_attack = './features_novas/' + 'features_attack' + '.arff'

features_train = './features_novas/' + 'features_train' + '.arff'
features_test = './features_novas/' + 'features_test' + '.arff'

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

    for i in range(qtd_attack_train):
        sorted = randint(0, len(vector_features_attack)-1)
        vector_features_train.append(vector_features_attack.pop(sorted))

    for i in range(qtd_normal_train):
        sorted = randint(0, len(vector_features_normal)-1)
        vector_features_train.append(vector_features_normal.pop(sorted))

    for i in range(len(vector_features_attack)):
        vector_features_test.append(vector_features_attack[i])

    for i in range(len(vector_features_normal)):
        vector_features_test.append(vector_features_normal[i])

    train.writelines(vector_features_train)
    test.writelines(vector_features_test)

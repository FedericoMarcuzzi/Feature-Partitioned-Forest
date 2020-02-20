'''
test.py

Created by 

name: Federico Marcuzzi
e-mail: federico.marcuzzi@unive.it

date 20/02/2020
'''

from misc import *
from models import *
from certificates import *
from attack_generator import *

from sklearn.ensemble import RandomForestClassifier

### PARAMETERS
n_threads = 8
b = 3
r = 4
n_est = r * (2 * b + 1)
random_state = 7
max_leaf_nodes = 8
max_attack = 2


### DATASET: breast_cancer, spam_base, wine
data_name = 'wine'
path = 'dataset/'

dataset, labels = load_dataset(path + data_name)
dataset = normalize(dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.34, random_state=random_state,shuffle=True,stratify=labels)

n_ist, _ = np.shape(X_test)


### FEATURE PARTITIONED FOREST
print('\n\nFEATURE PARTITIONED FOREST\n')
clf = FeaturePartitionedForest(b=b, r=r , min_acc=None,random_state=random_state, max_leaf_nodes=max_leaf_nodes)
clf.fit(X_train,y_train)

acc_train = clf.score(X_train,y_train)
acc_test = clf.score(X_test,y_test)

print('acc train: ',acc_train)
print('acc test: ',acc_test)

print('\nAccurate Lower-Bound')
acc_alb = np.round([1 - len(broken[1])/n_ist for broken in accurate_lower_bound(clf,X_test,y_test,1,b).items()], decimals=3)
print(acc_alb)

print('\nFast Lower-Bound')
acc_flb = np.round([1 - len(broken[1])/n_ist for broken in accurate_lower_bound(clf,X_test,y_test,1,b).items()], decimals=3)
print(acc_alb)

print()


acc_undr_atk = 1
for k in np.arange(1,max_attack+1):
    if acc_undr_atk != 0:
        acc_undr_atk = brute_force(clf, X_test, y_test, k, n_th=n_threads)
        print(acc_undr_atk,'                                                ')

print()


### RANDOM SUBSPACE METHOD
print('\n\nRANDOM SUBSPACE METHOD\n')
clf = RandomSubspaceMethod(p=0.2, n_trees=n_est, random_state=random_state, max_leaf_nodes=max_leaf_nodes)
clf.fit(X_train,y_train)

acc_train = clf.score(X_train,y_train)
acc_test = clf.score(X_test,y_test)

print('acc train: ',acc_train)
print('acc test: ',acc_test)

print('\nAccurate Lower-Bound')
acc_alb = np.round([1 - len(broken[1])/n_ist for broken in accurate_lower_bound(clf,X_test,y_test,1,b).items()], decimals=3)
print(acc_alb)

print('\nFast Lower-Bound')
acc_flb = np.round([1 - len(broken[1])/n_ist for broken in accurate_lower_bound(clf,X_test,y_test,1,b).items()], decimals=3)
print(acc_alb)

print()

acc_undr_atk = 1
for k in np.arange(1,max_attack+1):
    if acc_undr_atk != 0:
        acc_undr_atk = brute_force(clf, X_test, y_test, k, n_th=n_threads)
        print(acc_undr_atk,'                                                ')

print('\n')

### RANDOM FOREST
print('\n\nRANDOM FOREST\n')
clf = RandomForestClassifier(n_estimators=n_est, random_state=random_state, max_leaf_nodes=max_leaf_nodes)
clf.fit(X_train,y_train)

acc_train = clf.score(X_train,y_train)
acc_test = clf.score(X_test,y_test)

print('acc train: ',acc_train)
print('acc test: ',acc_test)

n_ist, _ = np.shape(X_test)

print('Accurate Lower-Bound')
acc_alb = np.round([1 - len(broken[1])/n_ist for broken in accurate_lower_bound(clf,X_test,y_test,1,b).items()], decimals=3)
print(acc_alb)

print('Fast Lower-Bound')
acc_flb = np.round([1 - len(broken[1])/n_ist for broken in accurate_lower_bound(clf,X_test,y_test,1,b).items()], decimals=3)
print(acc_alb)

acc_undr_atk = 1
for k in np.arange(1,max_attack+1):
    if acc_undr_atk != 0:
        acc_undr_atk = brute_force(clf, X_test, y_test, k, n_th=n_threads)
        print(acc_undr_atk,'                                                ')
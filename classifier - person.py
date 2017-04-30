
import pandas as pd
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import seaborn as sns

def ML_test(learn_train, learn_test, train_class, test_class, n):
    
    accu_cols=["Classifier", "Accuracy"]
    accu_bar = pd.DataFrame(columns=accu_cols)


    in_class_accuracy = [pd.DataFrame(columns = [str(i), 'accuracy'] ) for i in range(n)]

    #Part 1： SVM
    from sklearn.svm import SVC
    svm_clf = SVC()
    svm_clf = svm_clf.fit(learn_train, train_class)
    accu = svm_clf.score(learn_test ,test_class)
    print ('SVM: ', accu) 
    accu_entry = pd.DataFrame([['SVM', accu*100]], columns=accu_cols)
    accu_bar = accu_bar.append(accu_entry)

    for class_label in range(n):
        in_class_test = [learn_test[i] for i, k in enumerate(test_class) if k == class_label]
        in_class_test = np.asarray(in_class_test)
        if len(in_class_test) == 0:
            accu = 1.0
        else:
            accu = svm_clf.score(in_class_test, np.asarray([class_label for i in range(len(in_class_test))]))
        accu_entry = pd.DataFrame([['SVM', accu*100]], columns = [str(class_label), 'accuracy'])
        in_class_accuracy[class_label] = in_class_accuracy[class_label].append(accu_entry) 


    #Part 2: Decision Tree
    from sklearn import tree 
    tree_clf = tree.DecisionTreeClassifier(max_depth = 6)
    tree_clf = tree_clf.fit(learn_train, train_class)
    accu = tree_clf.score(learn_test ,test_class)
    print ('Decision Tree: ', accu) 
    accu_entry = pd.DataFrame([['Decision Tree', accu*100]], columns=accu_cols)
    accu_bar = accu_bar.append(accu_entry)
    
    for class_label in range(n):
        in_class_test = [learn_test[i] for i, k in enumerate(test_class) if k == class_label]
        in_class_test = np.asarray(in_class_test)
        if len(in_class_test) == 0:
            accu = 1.0
        else:
            accu = tree_clf.score(in_class_test, np.asarray([class_label for i in range(len(in_class_test))]))
        accu_entry = pd.DataFrame([['Decision Tree', accu*100]], columns = [str(class_label), 'accuracy'])
        in_class_accuracy[class_label] = in_class_accuracy[class_label].append(accu_entry) 

    #Part 2.1: Decision Forest
    from sklearn.ensemble import RandomForestClassifier
    forest_clf = RandomForestClassifier(max_depth = 5)
    forest_clf = forest_clf.fit(learn_train, train_class)
    accu = forest_clf.score(learn_test ,test_class)
    print ('Decision Forest: ', accu) 
    accu_entry = pd.DataFrame([['Decision Forest', accu*100]], columns=accu_cols)
    accu_bar = accu_bar.append(accu_entry)
    
    for class_label in range(n):
        in_class_test = [learn_test[i] for i, k in enumerate(test_class) if k == class_label]
        in_class_test = np.asarray(in_class_test)
        if len(in_class_test) == 0:
            accu = 1.0
        else:
            accu = forest_clf.score(in_class_test, np.asarray([class_label for i in range(len(in_class_test))]))
        accu_entry = pd.DataFrame([['Decision Forest', accu*100]], columns = [str(class_label), 'accuracy'])
        in_class_accuracy[class_label] = in_class_accuracy[class_label].append(accu_entry) 



    #Part 3: AdaBoost
    from sklearn.ensemble import AdaBoostClassifier
    adaboost_clf = AdaBoostClassifier()
    adaboost_clf = adaboost_clf.fit(learn_train, train_class)
    accu = adaboost_clf.score(learn_test ,test_class)
    print ('Adaboost: ', accu)
    accu_entry = pd.DataFrame([['Adaboost', accu*100]], columns=accu_cols)
    accu_bar = accu_bar.append(accu_entry)
    
    for class_label in range(n):
        in_class_test = [learn_test[i] for i, k in enumerate(test_class) if k == class_label]
        in_class_test = np.asarray(in_class_test)
        if len(in_class_test) == 0:
            accu = 1.0
        else:
            accu = adaboost_clf.score(in_class_test, np.asarray([class_label for i in range(len(in_class_test))]))
        accu_entry = pd.DataFrame([['Adaboost', accu*100]], columns = [str(class_label), 'accuracy'])
        in_class_accuracy[class_label] = in_class_accuracy[class_label].append(accu_entry) 
 

    #Part 4: Neutral Network
    from sklearn.neural_network import MLPClassifier
    MLP_clf = MLPClassifier()
    MLP_clf = MLP_clf.fit(learn_train, train_class)
    accu = MLP_clf.score(learn_test ,test_class)
    print ('Neutral Network: ', accu)
    accu_entry = pd.DataFrame([['Neutral Network', accu*100]], columns=accu_cols)
    accu_bar = accu_bar.append(accu_entry)
    
    for class_label in range(n):
        in_class_test = [learn_test[i] for i, k in enumerate(test_class) if k == class_label]
        in_class_test = np.asarray(in_class_test)
        if len(in_class_test) == 0:
            accu = 1.0
        else:
            accu = MLP_clf.score(in_class_test, np.asarray([class_label for i in range(len(in_class_test))]))
        accu_entry = pd.DataFrame([['Neutral Network', accu*100]], columns = [str(class_label), 'accuracy'])
        in_class_accuracy[class_label] = in_class_accuracy[class_label].append(accu_entry) 


    #Part 5: Gaussian
    from sklearn.naive_bayes import GaussianNB
    Gauss_clf = GaussianNB()
    Gauss_clf = Gauss_clf.fit(learn_train, train_class)
    accu = Gauss_clf.score(learn_test ,test_class)
    print ('Naive Bayes: ', accu)
    accu_entry = pd.DataFrame([['Naive Bayes', accu*100]], columns=accu_cols)
    accu_bar = accu_bar.append(accu_entry)
    
    for class_label in range(n):
        in_class_test = [learn_test[i] for i, k in enumerate(test_class) if k == class_label]
        in_class_test = np.asarray(in_class_test)
        if len(in_class_test) == 0:
            accu = 1.0
        else:
            accu = Gauss_clf.score(in_class_test, np.asarray([class_label for i in range(len(in_class_test))]))
        accu_entry = pd.DataFrame([['Naive Bayes', accu*100]], columns = [str(class_label), 'accuracy'])
        in_class_accuracy[class_label] = in_class_accuracy[class_label].append(accu_entry) 
 

    #Part 6: Voting Classifier
    from sklearn.ensemble import VotingClassifier
    svm_clf = SVC(probability= True)
    vote_clf = VotingClassifier(estimators=[('svm', svm_clf), ('nw', MLP_clf), ('dt', tree_clf)])
    vote_clf = vote_clf.fit(learn_train, train_class)
    accu = vote_clf.score(learn_test ,test_class)
    print ('Voting Classifier: ', accu)
    accu_entry = pd.DataFrame([['Voting Classifier', accu*100]], columns=accu_cols)
    accu_bar = accu_bar.append(accu_entry)
    
    sns.barplot(x='Accuracy', y='Classifier', data=accu_bar)
    plt.xlabel('Accuracy %')
    #plt.title('Classifier Accuracy')
    plt.show()

    for class_label in range(n):
        in_class_test = [learn_test[i] for i, k in enumerate(test_class) if k == class_label]
        in_class_test = np.asarray(in_class_test)
        if len(in_class_test) == 0:
            accu = 1.0
        else:
            accu = vote_clf.score(in_class_test, np.asarray([class_label for i in range(len(in_class_test))]))
        accu_entry = pd.DataFrame([['Voting classifier', accu*100]], columns = [str(class_label), 'accuracy'])
        in_class_accuracy[class_label] = in_class_accuracy[class_label].append(accu_entry) 

    
    #Section 2.1 - Hybrid classifier 
    def hybrid_clf(X):
        Gauss_result = Gauss_clf.predict(X)
        adaboost_result = adaboost_clf.predict(X)
        hybrid_result = []
        for i in range(len(X)):
           hybrid_result.append(max(Gauss_result[i], adaboost_result[i]))
        return hybrid_result
        
    if n == 2:
        hybrid_predict_result = hybrid_clf(learn_test)
        templist = [hybrid_predict_result[i] == k for i,k in enumerate(test_class)]
        accu = templist.count(True)/len(templist)
        print ('Hybrid Classifier: ', accu)

        
        
    #Section 2:
    
    #Part 1: accu vs max_depth in decision tree
    accu_record = list()
    x_list = [i+1 for i in range(20)]
    for i in x_list:
        tree_clf = tree.DecisionTreeClassifier(max_depth = i)
        tree_clf = tree_clf.fit(learn_train, train_class)
        accu = tree_clf.score(learn_test ,test_class)
        accu_record.append(accu)
    
    #generate graph
    plt.plot(x_list, accu_record)
    #plt.title('Accuracy vs depth of decision tree')
    plt.show()
    
    #Part 2: accu vs hidden layers of neutral network
    accu_record = []
    x_list = [10*i+30 for i in range(20)]
    for i in x_list:
        MLP_clf = MLPClassifier(hidden_layer_sizes = i)
        MLP_clf = MLP_clf.fit(learn_train, train_class)
        accu = MLP_clf.score(learn_test ,test_class)
        accu_record.append(accu)
    
    #generate graph
    plt.plot(x_list, accu_record)
    #plt.title('Accuracy vs number of hidden layers in neutral network')
    plt.show()
    
    #Part 3: accu vs number of weak classifiers in ababoost 
    accu_record = []
    x_list = [5*(i+1) for i in range(20)]
    for i in x_list:
        adaboost_clf = AdaBoostClassifier(n_estimators=i)
        adaboost_clf = adaboost_clf.fit(learn_train, train_class)
        accu = adaboost_clf.score(learn_test ,test_class)                
        accu_record.append(accu)
    
    #generate graph
    plt.plot(x_list, accu_record)
    #plt.title('Accuracy vs number of weak classifiers in AdaBoost')
    plt.show()
    
    
    return in_class_accuracy

dataset_dir = "2015-traffic-fatalities\\"

person_data = pd.read_csv(dataset_dir + "person.csv")
#CLEAN DATA:

delete_label =['ST_CASE', 'STATE', 'COUNTY', 'MONTH', 'HOUR', 'MINUTE', 'DAY', 'DEATH_DA', 'DEATH_MO', 'DEATH_YR', 'DEATH_HR', 'DEATH_MN', 'DEATH_TM', 'LAG_HRS', 'LAG_MINS']
delete_label = delete_label + ['WORK_INJ','HISPANIC', 'RACE', 'LOCATION', 'VEH_NO', 'PER_NO', 'DOA', 'P_SF1', 'P_SF2', 'P_SF3', 'STR_VEH', 'EXTRICAT', 'ROLLOVER']
delete_label = delete_label + ['MAK_MOD', 'MAKE', 'MOD_YEAR', 'ALC_RES', 'DRUGRES1', 'DRUGRES2', 'DRUGRES3', 'ATST_TYP', 'DRUG_DET', 'EMER_USE', 'ALC_STATUS', 'SPEC_USE']
delete_label = delete_label + ['DSTATUS', 'HOSPITAL', 'EJ_PATH', 'EJECTION', 'FIRE_EXP', 'HARM_EV', 'MAN_COLL', 'FUNC_SYS', 'RUR_URB', 'VE_FORMS', 'IMPACT1']
for i in delete_label:
    del person_data[i]

person_data = person_data[person_data.AGE < 98]
person_data = person_data[person_data.SEX < 8]
person_data = person_data[person_data.DRUGS < 8]
person_data = person_data[person_data.DRINKING < 8]
person_data = person_data[person_data.AIR_BAG < 8]
person_data = person_data[person_data.INJ_SEV < 8]
person_data = person_data[person_data.PER_TYP < 9]
person_data = person_data[person_data.SEAT_POS < 98]
person_data = person_data[person_data.REST_USE < 98]
person_data = person_data[person_data.ALC_DET < 9]
person_data = person_data[person_data.TOW_VEH < 9]

#MACHINE LEARNING PART

feature_set = person_data.sample(frac = 1)
#split 1/4 as test set, 3/4 as train set
partition_point = round(len(feature_set)/4)
learn_test = feature_set[0:partition_point].copy()
learn_train = feature_set[partition_point:].copy()
test_class = learn_test['INJ_SEV']  
train_class = learn_train['INJ_SEV']

"""
Distribution of classes
"""
dist_cols=['label', 'frequency']
dist_bar = pd.DataFrame(columns= dist_cols)
class_info = feature_set['INJ_SEV'].as_matrix()
class_info = class_info.tolist()

label = ['No injury', 'Possible injury', 'Suspected minor injury', 'Suspected serious injury', 'Fatal injury', 'Injured - severity unknown', 'Death']

for i in set(class_info):
    rate = class_info.count(i)*100/len(class_info)
    dist_entry = pd.DataFrame([[str(i), rate]], columns=dist_cols)
    dist_bar = dist_bar.append(dist_entry)

sns.barplot(y='frequency', x='label', data=dist_bar)
plt.xlabel('Frequency %')
#plt.title('Distribution of classes')
plt.show()


"""
TEST 1: 7 different classes: 0 - 6
"""

del learn_test['INJ_SEV']
del learn_train['INJ_SEV']
learn_train = learn_train.as_matrix() #ndarray
train_class = train_class.as_matrix()
test_class = test_class.as_matrix()
learn_test = learn_test.as_matrix() #ndarray

train_index = list(range(len(train_class)))

test1_result = ML_test(learn_train, learn_test, train_class, test_class, 7)


"""
TEST 2: 2 different classes: 0 vs 1-6 (No injury vs Injury)
"""

def binary_convert(x):
    if x > 0:
        return 1
    else:
        return 0

test_class = [binary_convert(i) for i in test_class]
train_class = [binary_convert(i) for i in train_class]
test_class = np.asarray(test_class)
train_class = np.asarray(train_class)

test2_result = ML_test(learn_train, learn_test, train_class, test_class, 2)

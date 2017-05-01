
import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
from subprocess import check_output


# filter out unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


data = pd.read_csv('/Users/whglamrock/anaconda/lib/python2.7/site-packages/accident.csv')
import sklearn.utils
data = sklearn.utils.shuffle(data)

states = {1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE',
          11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS',
          21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN',
          28:'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH', 34: 'NJ', 35: 'NM',
          36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK', 41: 'OR', 42: 'PN', 43: 'PR',
          44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 52: 'VI',
          53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY'}


#fig, axes = plt.subplots(nrows = 4, ncols = 1,figsize = (8, 8))
#fig.subplots_adjust(hspace = 0.8)

data['state'] = data['STATE'].apply(lambda x: states[x])
Total_ac = data['state'].value_counts()
#data['state'].value_counts().plot(ax = axes[0], kind = 'bar',title = 'state-wise accidents')

df_drinking = pd.concat([data['state'], data['DRUNK_DR']], axis = 1)
drk_state = df_drinking.groupby('state')
#drk_state.sum().sort_index(by = 'DRUNK_DR', ascending = False).plot(ax = axes[1], kind = 'bar',title = 'state-wise drunk drivers')


Total_ac.sort_index(ascending = True)
drk_break = pd.concat([Total_ac.sort_index(ascending = True),drk_state.sum()], axis = 1)
drk_break.columns = ['People_involved', 'Drunk_drivers']
drk_break['NoN Drinking individuals'] = drk_break['People_involved'] - drk_break['Drunk_drivers']
#drk_break[['NoN Drinking individuals','Drunk_drivers']].sort_index(by = 'NoN Drinking individuals',ascending = False).plot.bar(ax = axes[2],stacked = 'True')


drk_break['Drunk_Dr_per_population'] = drk_break['Drunk_drivers']/drk_break['People_involved']
drk_break.head()
#drk_break[['Drunk_Dr_per_population']].sort_index(by = 'Drunk_Dr_per_population', ascending = False).plot(ax = axes[3],kind = 'bar')


month = {1: '1jan', 2: '2feb', 3: '3mar', 4: '4april', 5: '5may', 6: '6june', 7: '7july', 8: '8aug', 9: '90sep', 10: '91oct',11: '92nov',12: '93dec'}

# set "nrows = 1, ncols = 1" when plotting the four demographic pictures
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (9, 6))
fig.subplots_adjust(hspace = .6)

##### count by month
data['month'] = data['MONTH'].apply(lambda x: month[x])
#data['month'].value_counts().sort_index(level = 'month').plot(ax = axes[0,0], kind = 'bar', title = 'Month-wise accident')

### count by month day
#data['DAY'].value_counts().sort_index().plot(ax = axes[0,1], title = 'Day-wise accident')

####### week day count
day = {1: '1_SAT', 2: '2_SUN', 3: '3_MON', 4: '4_TUE', 5: '5_WED', 6: '6_THU', 7: '7_FRI'}
data['day_week'] = data['DAY_WEEK'].apply(lambda x: day[x])
#data['day_week'].value_counts().sort_index().plot(ax = axes[1,0], kind = 'bar', title = 'Week_Day-wise accident')

############# count by hour
data = data[data.HOUR != 99]
#data['HOUR'].value_counts().sort_index().plot(ax = axes[1,1], kind = 'bar', title = 'Hour-wise accident')


data['HARM_EV'].value_counts().head()
harm_ev = {12: 'SameRoadVehicle', 8: 'Pedestrian', 1: 'OverTurn', 42: 'Trees', 33: 'Curb', 34: 'Ditch', 35: 'Embankment'}
data['harm_ev'] = data['HARM_EV'].apply(lambda x: harm_ev[x] if (x == 12 or x == 8 or x == 1 or x == 42 or x == 33 or x == 34 or x == 35) else 'Other')
#data['harm_ev'].value_counts().plot(kind = 'pie', title = 'How harmful environment played role')


df_ml = data[['state','MONTH','DAY_WEEK','DAY','HOUR','harm_ev','DRUNK_DR']]
df_ml['state'] = df_ml['state'].fillna('TX')
df_ml['MONTH'] = df_ml['MONTH'].fillna(0)
df_ml['DAY_WEEK'] = df_ml['DAY_WEEK'].fillna(6)
df_ml['DAY'] = df_ml['DAY'].fillna(3)
df_ml['HOUR'] = df_ml['HOUR'].fillna(18)
df_ml['harm_ev'] = df_ml['harm_ev'].fillna('Embankment')
df_ml['DRUNK_DR'] = df_ml['DRUNK_DR'].fillna(0)
df_ml['harm_ev'].unique()

df_ml['harm_ev'] = df_ml['harm_ev'].replace(['Embankment', 'SameRoadVehicle'], ['EM', 'SRV'])
x = pd.get_dummies(df_ml['harm_ev'], prefix = 'harm_ev')
x = pd.concat([x, pd.get_dummies(df_ml['state'], prefix ='state')], axis = 1)
x = pd.concat([x, pd.get_dummies(df_ml['MONTH'], prefix ='MONTH')], axis = 1)
x = pd.concat([x, pd.get_dummies(df_ml['DAY_WEEK'], prefix ='Dw')], axis = 1)
x = pd.concat([x, pd.get_dummies(df_ml['DAY'], prefix ='DAY')], axis = 1)
x = pd.concat([x, pd.get_dummies(df_ml['HOUR'], prefix ='HOUR')], axis = 1)

df_ml['DRUNK_DR'] = df_ml['DRUNK_DR'].apply(lambda x: 0 if x == 0 else 1)
df_ml['DRUNK_DR'].value_counts()

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y = le.fit_transform(df_ml['DRUNK_DR'])
le.inverse_transform([0, 1])
dt = DecisionTreeClassifier(max_depth = 4)
dt.fit(x.values, Y)

from sklearn import cross_validation
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)


s = []
for i in range(13):
   s.append(0)

dt = DecisionTreeClassifier(max_depth = 1,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
i = 0
s[i] = scores.mean()
i = i + 1
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

dt = DecisionTreeClassifier(max_depth = 2,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 3,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 4,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 5,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 6,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 7,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 8,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 9,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 10,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 12,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 14,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 16,criterion='entropy')
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()

n = [1,2,3,4,5,6,7,8,9,10,12,14,16]

#plt.plot(n, s, 'r', lw=3)
#plt.title('Accuracy vs Depth of Decision Tree with entropy as criterion')


s = []
for i in range(10):
   s.append(0)

dt = DecisionTreeClassifier(max_depth = 1)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
i = 0
s[i]=scores.mean()
i = i + 1
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

dt = DecisionTreeClassifier(max_depth = 2)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 3)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 4)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 5)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 6)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 7)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 8)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 9)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 10)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()

#plt.plot(s, 'r', lw=3)
#plt.title('Accuracy vs Depth of Decision Tree with information gain as leafing criterion')

data['MAN_COLL'].value_counts()
man_coll = {0: 'NoCol',6: 'angle',2: 'headOn',1: 'Rear',7: 'sideswipe'}
data['man_coll'] = data['MAN_COLL'].apply(lambda x: man_coll[x] if (x == 0 or x == 6 or x == 2 or x == 1 or x == 7)  else 'NoCol')
#data['man_coll'].value_counts().plot(kind = 'pie',title = 'Manner of collisions')


x = pd.concat([x, pd.get_dummies(data['man_coll'], prefix ='man_coll')], axis=1)

s = []
for i in range(10):
   s.append(0)

dt = DecisionTreeClassifier(max_depth = 1)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
i = 0
s[i] = scores.mean()
i = i + 1
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

dt = DecisionTreeClassifier(max_depth = 2)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i]=scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 3)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i]=scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 4)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i]=scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 5)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i]=scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 6)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i]=scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 7)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i]=scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 8)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i]=scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 9)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i]=scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 10)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()


data['WEATHER'].value_counts()
weather = {1: 'clear', 10: 'couldy', 2: 'rain', 5: 'fog', 4: 'snow', 99: 'unknown', 3: 'sleet', 98: 'unreported', 8: 'other', 12: 'drizzle', 11: 'blowingSnow', 6: 'crosswinds', 7: 'blowingSand'}
data['weather'] = data['WEATHER'].apply(lambda x: weather[x] )
#data['weather'].value_counts().plot.bar(figsize = (8, 4))


x = pd.concat([x, pd.get_dummies(data['weather'], prefix ='weather')], axis = 1)

s = []
for i in range(10):
   s.append(0)

dt = DecisionTreeClassifier(max_depth = 1)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
i = 0
s[i] = scores.mean()
i = i + 1
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

dt = DecisionTreeClassifier(max_depth = 2)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 3)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 4)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 5)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 6)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 7)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 8)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 9)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()
i = i + 1

dt = DecisionTreeClassifier(max_depth = 10)
scores = cross_validation.cross_val_score(dt, x, Y, cv = 10)
#print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
s[i] = scores.mean()


n = 3000

xx = x.iloc[0:n]
label = df_ml['DRUNK_DR'].iloc[0:n]

msk = np.random.rand(len(xx)) < 0.8
train_f = xx[msk]
test_f = xx[~msk]
train_l = label[msk]
test_l = label[~msk]

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel = "rbf", C = 0.025, probability = True),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(train_f, train_l)
    name = clf.__class__.__name__

    print("="*30)
    print(name)

    print('****Results****')
    train_predictions = clf.predict(test_f)
    acc = accuracy_score(test_l, train_predictions)
    print("Accuracy: {:.4%}".format(acc))

    train_predictions = clf.predict_proba(test_f)
    ll = log_loss(test_l, train_predictions)
    print("Log Loss: {}".format(ll))

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)

print("="*30)

import seaborn as sns
#sns.set_color_codes("muted")
#sns.barplot(x='Accuracy', y='Classifier', data=log)

#plt.xlabel('Accuracy %')
#plt.title('Classifier Accuracy')

sns.barplot(x = 'Log Loss', y = 'Classifier', data = log)

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()

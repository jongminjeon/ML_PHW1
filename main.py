import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Standard Scaling Module
def StandardScaling(data):
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_df)
    return scaled_df

# MinMax Scaling Module
def MinMaxScaling(data):
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_df)
    return scaled_df

# Robust Scaling Module
def RobustScaling(data):
    scaler = preprocessing.RobustScaler()
    scaled_df = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_df)
    return scaled_df

# Method to Split Train and Test dataset.
def SplitData(X, Y, testsize):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize, shuffle=False, random_state=0)
    return X_train, X_test, Y_train, Y_test

# Preprocessing
def Preprocessing(df, Encode, Scaling):         # Encode = 0 -> Use Ordinal Encoder. If Encode = 2, Use Label Encoder.
    df.replace({"?": np.nan}, inplace=True)     # Scaling = 0 -> Standard Scaler, Scaling = 1 -> MinMax Scaler, Scaling = 3 -> Robust Scaler
    df.dropna(axis=0, inplace=True)
    df = df.drop(columns=['ID'])
    encode1 = preprocessing.OrdinalEncoder()
    encode2 = preprocessing.LabelEncoder()
    if Encode == 0 and Scaling == 0 :
        encode1.fit(df['Bare_N'][:, np.newaxis])
        df['Bare_N'] = encode1.transform(df['Bare_N'][:, np.newaxis]).reshape(-1)
        preprocessed_data_list.append(StandardScaling(df))
    if Encode == 0 and Scaling == 1 :
        encode1.fit(df['Bare_N'][:, np.newaxis])
        df['Bare_N'] = encode1.transform(df['Bare_N'][:, np.newaxis]).reshape(-1)
        preprocessed_data_list.append(MinMaxScaling(df))
    if Encode == 0 and Scaling == 2 :
        encode1.fit(df['Bare_N'][:, np.newaxis])
        df['Bare_N'] = encode1.transform(df['Bare_N'][:, np.newaxis]).reshape(-1)
        preprocessed_data_list.append(RobustScaling(df))
    if Encode == 1 and Scaling == 0 :
        encode2.fit(df['Bare_N'][:, np.newaxis])
        df['Bare_N'] = encode2.transform(df['Bare_N'][:, np.newaxis]).reshape(-1)
        preprocessed_data_list.append(StandardScaling(df))
    if Encode == 1 and Scaling == 1 :
        encode2.fit(df['Bare_N'][:, np.newaxis])
        df['Bare_N'] = encode2.transform(df['Bare_N'][:, np.newaxis]).reshape(-1)
        preprocessed_data_list.append(MinMaxScaling(df))
    if Encode == 1 and Scaling == 2 :
        encode2.fit(df['Bare_N'][:, np.newaxis])
        df['Bare_N'] = encode2.transform(df['Bare_N'][:, np.newaxis]).reshape(-1)
        preprocessed_data_list.append(RobustScaling(df))

def Make_Combination(df):
    Preprocessing(df, 0, 0)
    Preprocessing(df, 0, 1)
    Preprocessing(df, 0, 2)
    Preprocessing(df, 1, 0)
    Preprocessing(df, 1, 1)
    Preprocessing(df, 1, 2)

######### Model Building ###########
def DecisionTree_Entropy(CrossVal):
    Tree = DecisionTreeClassifier(criterion='entropy')
    Tree_Entropy_grid = {
        'max_depth': [2, 3, 4, 5, 6],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3, 4]
    }
    gsTree_Entropy = GridSearchCV(Tree, param_grid=Tree_Entropy_grid, cv=CrossVal, refit=True)
    gsTree_Entropy.fit(X_train, Y_train)
    TestSetScore_list.append(gsTree_Entropy.score(X_test, Y_test))
    Best_Parameters_list.append(gsTree_Entropy.best_params_)
    Best_Accuracy_list.append(gsTree_Entropy.best_score_)
    Model_list.append('Decision Tree - entropy')
    print('DecisionTreeClassifier(Entropy) TestSet Score:', gsTree_Entropy.score(X_test, Y_test))
    print('DecisionTreeClassifier(Entropy) Best Parameter:', gsTree_Entropy.best_params_)
    print('DecisionTreeClassifier(Entropy) Best Accuracy: {0:.4f}'.format(gsTree_Entropy.best_score_))

def DecisionTree_Gini(CrossVal):
    Tree = DecisionTreeClassifier(criterion='gini')
    Tree_Gini_grid = {
        'max_depth': [2, 3, 4, 5, 6],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3, 4]
    }
    gsTree_Gini = GridSearchCV(Tree, param_grid=Tree_Gini_grid, cv=CrossVal, refit=True)
    gsTree_Gini.fit(X_train, Y_train)
    TestSetScore_list.append(gsTree_Gini.score(X_test, Y_test))
    Best_Parameters_list.append(gsTree_Gini.best_params_)
    Best_Accuracy_list.append(gsTree_Gini.best_score_)
    Model_list.append('DecisionTree - Gini')
    print('DecisionTreeClassifier(Gini) TestSet Score:', gsTree_Gini.score(X_test, Y_test))
    print('DecisionTreeClassifier(Gini) Best Parameter:', gsTree_Gini.best_params_)
    print('DecisionTreeClassifier(Gini) Best Accuracy: {0:.4f}'.format(gsTree_Gini.best_score_))

def LogisticReg(CrossVal):
    reg = LogisticRegression()
    reg_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2']
    }
    regCV = GridSearchCV(reg, param_grid=reg_grid, cv=CrossVal, refit=True)
    regCV.fit(X_train, Y_train)
    TestSetScore_list.append(regCV.score(X_test, Y_test))
    Best_Parameters_list.append(regCV.best_params_)
    Best_Accuracy_list.append(regCV.best_score_)
    Model_list.append('Logistic Regression')
    print('Logistic Regression TestSet Score:', regCV.score(X_test, Y_test))
    print('Logistic Regression Best Parameter:', regCV.best_params_)
    print('Logistic Regression Best Accuracy: {0:.4f}'.format(regCV.best_score_))

def SVM(CrossVal):
    svm = SVC()
    svm_grid = {
        'C': [0.1, 1, 10],
        'gamma': [ 1, 0.1, 0.01],
        'kernel': ['linear', 'rbf']
    }
    svmCV = GridSearchCV(svm, param_grid=svm_grid, cv=CrossVal, refit=True)
    svmCV.fit(X_train, Y_train)
    TestSetScore_list.append(svmCV.score(X_test, Y_test))
    Best_Parameters_list.append(svmCV.best_params_)
    Best_Accuracy_list.append(svmCV.best_score_)
    Model_list.append('Support Vector Machine')
    print('Support Vector Machine TestSet Score:', svmCV.score(X_test, Y_test))
    print('Support Vector Machine Best Parameter:', svmCV.best_params_)
    print('Support Vector Machine Best Accuracy: {0:.4f}'.format(svmCV.best_score_))

# Load Data
cols = ['ID', 'Thickness', 'Cell_size', 'Cell_shape', 'Adhesion', 'Epi_Cell_size', 'Bare_N', 'Bland', 'Normal_N',
        'Mitoses', 'Class']
data = pd.read_csv('breast_cancer_wisconsin.data', sep=',', encoding='cp949', names=cols)
data = pd.DataFrame(data)

# Define Global Variable
preprocessed_data_list = []
Make_Combination(data)
TestSetScore_list = []      # Store scores in the list
Best_Parameters_list = []   # Store Model Best parameters in the list
Best_Accuracy_list = []     # Store Model Best accuracy in the list
Model_list = []             # Store which model was used
Dataset_Num = 0

# For each Preprocessed Datasets, Fit four models and store the scores.
for i in preprocessed_data_list:
    data = preprocessed_data_list[Dataset_Num]
    data = data.astype('int')
    print('\n###########', Dataset_Num + 1, 'th Dataset ############')
    X = data.iloc[:,0:-1]    # When you use in the other dataset, Change the location of the target.
    Y = data.iloc[:,-1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=0)
    DecisionTree_Entropy(10)    # K-Fold Cross Validation
    DecisionTree_Gini(10)       # K-Fold Cross Validation
    SVM(10)                     # K-Fold Cross Validation
    LogisticReg(10)             # K-Fold Cross Validation
    Dataset_Num = Dataset_Num + 1

print('The Best Score among the combinations : ', max(TestSetScore_list))
print('The Parameters of Best Score : ', Best_Parameters_list[TestSetScore_list.index(max(TestSetScore_list))])
print('The Best Accuracy : ', Best_Accuracy_list[TestSetScore_list.index(max(TestSetScore_list))])
print('Used Model : ', Model_list[TestSetScore_list.index(max(TestSetScore_list))])

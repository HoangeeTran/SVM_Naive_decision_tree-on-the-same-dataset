"""
SVM, logistic, naive, decision tree, random forest model in solving classification problem.
The dataset you will be working on is 'data-breast-cancer.csv'. It is composed of attributes to build a prediction model.
This is a dataset used to detect whether a patient has breast cancer depending on the following features: 

- diagnosis: (label) the diagnosis of breast (label) tissues (M = malignant, B = benign).
    - radius: distances from center to points on the perimeter.
- texture: standard deviation of gray-scale values.
- perimeter: perimeter of the tumor.
- area: area of the tumor.
- smoothness: local variation in radius lengths.
- compactness: is equal to (perimeter^2 / area - 1.0).
- concavity: severity of concave portions of the contour.
- concave points: number of concave portions of the contour.
- symmetry: symmetry of the tumor shape.
- fractal dimension: "coastline approximation" - 1.
"""

# Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv("data-breast-cancer.csv")

###  exploratory data analysis 
print("the first 5 data samples:\n", df.head(5))
# Show data information
print("Show data information:")
print(df.info())
print("shape:")
print(df.shape)
# remove the first unnamed column that is used for indexing. It is not valuable for trainning the model.
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# show df
df.hist(figsize=(15, 15))
plt.show()
#Show Statistics of numerical values
print("Statistics of numerical values \n")
print(df.describe())
print()
print("Statistics of numerical and categorical values \n")
print(df.describe(include=[object, float]))

# print number of continuous columns and its name.
print("Continuous Columns")
continuous_columns = df.describe().columns
print(continuous_columns)

print("number of continuous columns")
print(len(continuous_columns))
### number check if the data is imbalanced or not
print(df["diagnosis"].value_counts())
#the data is not much imbalanced.

# convert categorical  variables to numerical for data label (diagnoses)
df["diagnosis"].replace(['M', 'B'],
                        [1, 0], inplace=True)
print("data samples after encoding:")
print(df.head())
# remove duplicate if any
df = df.drop_duplicates(ignore_index=True)     



### remove outliers and clean data

# Let us take the whisker as :The 2nd percentile and the 98th percentile of the data" to remove data outliers.

# Remove outliers of 'radius_mean' feature
for col in df.columns:
    q = df[col].quantile(0.98)  # Select q range as 98%
    df_clean = df[df[col] < q]
print("the shape of data after removing outliers:")
print(df_clean.shape)

# Note: this dataset does not contains Null values so we do not need to handle null value.
### preprocessing 
# Separate data features by removing the data label.
X = df_clean.drop(columns=["diagnosis"], axis=1)

# Assign data label to variable y
y = df_clean["diagnosis"]
print(y.shape)
# Split train/test with a random state
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, train_size=0.8)

# Initialize and use StandardScaler to normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalized_train = scaler.fit_transform(X_train)     # Fit and transform thr training data
X_normalized_test = scaler.transform(X_test)           # Only transform the test data.
feature = scaler.transform(X)

# Impport libraries to calculate evaluation metrics: precision, recall, f1 score.
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score

### Trainning the SVM model 
def SVM_model_trainning():    
    from sklearn.svm import SVC
    model = SVC()
    # apply cross-validation to count accuracy.
    cv_scores = cross_val_score(model, feature, y, cv=5)
    print(model, ' mean accuracy of 5-fold cross validation: ', cv_scores.mean())

    model.fit(X_normalized_train, y_train)

    # Show evaluation metrics on the test set
    predicted_label = model.predict(X_normalized_test)
    print("scores before tuning:")
    print("precision: ", precision_score(y_test, predicted_label))
    print("recall:", recall_score(y_test, predicted_label))
    print("f1", f1_score(y_test, predicted_label))
    print("Accuracy:", accuracy_score(y_test, predicted_label))
    print(classification_report(y_test, predicted_label))
    ###Model tuning

    param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000],
                "gamma": ["scale", 0.001, 0.005, 0.1]}
    gridsearch = GridSearchCV(SVC(), param_grid, cv=10, scoring="f1", verbose=1)     # cv: number of folds in cross validation.
    # Run grid search to find the best set of hyper-parameters
    gridsearch.fit(X_normalized_train, y_train)
    # print the best hyper-parameter
    print("The best hyper-parameter:\n", gridsearch.best_params_)
    # Re-run SVM with the best set of hyper-parameters.
    model = SVC(C=gridsearch.best_params_['C'], gamma=gridsearch.best_params_['gamma'])
    model.fit(X_normalized_train, y_train)
    # Show evaluation metrics on the test set
    predicted_label = model.predict(X_normalized_test)
    #print("scores after tuning:")
    print("precision: ", precision_score(y_test, predicted_label))
    print("recall:", recall_score(y_test, predicted_label))
    print("f1", f1_score(y_test, predicted_label))
    print("Accuracy:", accuracy_score(y_test, predicted_label))
    print(classification_report(y_test, predicted_label))
    print("in conclusion, the accuracy score before tuning and after tuning is the same, which is 0.964.the mean accuracy of validation is quite closed to accuracy on the test set.")
    

### Train the naive model 
def naive_model_trainning():
    ### train naive bayse model by gaussianNB

    from sklearn.naive_bayes import GaussianNB
    naive_model = GaussianNB()

    # apply cross-validation to count accuracy.
    cv_scores = cross_val_score(naive_model, feature, y, cv=5)
    print(naive_model, ' mean accuracy of 5-fold cross validation: ', cv_scores.mean())
    naive_model.fit(X_normalized_train, y_train)
    
    # Make prediction on the test data
    predicted_label = naive_model.predict(X_normalized_test)

    # Calculate evaluation metrics by comparing the prediction with the data label y_test
    print("precision: ", precision_score(y_test, predicted_label))
    print("recall:", recall_score(y_test, predicted_label))
    print("f1", f1_score(y_test, predicted_label))
    print("Accuracy:", accuracy_score(y_test, predicted_label))
    print(classification_report(y_test, predicted_label))
### train the logistic model
def logistic_model_trainning():
    from sklearn.linear_model import LogisticRegression
    logmodel = LogisticRegression()                      # Initialize Logistic Regression model
    # apply cross-validation to count accuracy.
    cv_scores = cross_val_score(logmodel, feature, y, cv=5)
    print(logmodel, ' mean accuracy of 5-fold cross validation: ', cv_scores.mean())
    logmodel.fit(X_normalized_train, y_train)
    
    # Make prediction on the test data
    predicted_label = logmodel.predict(X_normalized_test)

    # Calculate evaluation metrics by comparing the prediction with the data label y_test
    print("precision: ", precision_score(y_test, predicted_label))
    print("recall:", recall_score(y_test, predicted_label))
    print("f1", f1_score(y_test, predicted_label))
    print("Accuracy:", accuracy_score(y_test, predicted_label))
    print(classification_report(y_test, predicted_label))

### Train the decision tree model

def decision_tree_model_trainning():
    ### find the best hyper_parameter and trainning the model
    # Import GridSearchCV for finding the best hyper-parameter set.
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    params = {"criterion": ["gini", "entropy"],             # Criterion to evaluate the purity.
            "max_depth": [3, 5],                           # Maximum depth of the tree
            "min_samples_split": [4, 8]}                   # Stop splitting condition.

    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, cv=5)

    # Run the search on training data samples.
    grid_search.fit(X_train, y_train)
    # Best set of hyper-parameters found after searching
    print("the best set of hyper-parameters:\n", grid_search.best_params_)

    # Build a decision tree model pipeline from the best set of hyper-parameters found
    model_dt = DecisionTreeClassifier(criterion=grid_search.best_params_['criterion'], max_depth=grid_search.best_params_['max_depth'], min_samples_split=grid_search.best_params_['min_samples_split'])
    # Train the decision tree model
    model_dt.fit(X_normalized_train, y_train)
    # Make prediction on the test data
    predicted_label = model_dt.predict(X_normalized_test)

    # Calculate evaluation metrics by comparing the prediction with the data label y_test
    print("precision: ", precision_score(y_test, predicted_label))
    print("recall:", recall_score(y_test, predicted_label))
    print("f1", f1_score(y_test, predicted_label))
    print("Accuracy:", accuracy_score(y_test, predicted_label))
    print(classification_report(y_test, predicted_label))



    
### Building a RandomForest.
def random_forest_model_trainning():
    from sklearn.ensemble import RandomForestClassifier

    params = {"criterion": ["gini", "entropy"],             # Criterion to evaluate the purity.
            "max_depth": [7, 9, 11],                           # Maximum depth of the tree
            "min_samples_split": [8, 12, 16]}                   # Stop splitting condition.

    grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(n_estimators=10, n_jobs=10), param_grid=params, cv= 5) # Number of trees in the forest is 10

    # Run the search on training data samples.
    grid_search_rf.fit(X_train, y_train)     # Train the RandomForest
    # Best set of hyper parameters of the Random Forest
    print("Best set of hyper parameters of the Random Forest: \n", grid_search_rf.best_params_)

    # Build a Random Forest model pipeline from the best set of hyper-parameters found
    model_rf = RandomForestClassifier(n_estimators=10, random_state=1, criterion=grid_search_rf.best_params_['criterion'], max_depth=grid_search_rf.best_params_['max_depth'], min_samples_split=grid_search_rf.best_params_['min_samples_split'])     # Initialize the RandomForest
        # Train the random forest model
    model_rf.fit(X_normalized_train, y_train)
    # Make prediction on the test data
    predicted_label = model_rf.predict(X_normalized_test)

    # Calculate evaluation metrics by comparing the prediction with the data label y_test
    print("precision: ", precision_score(y_test, predicted_label))
    print("recall:", recall_score(y_test, predicted_label))
    print("f1", f1_score(y_test, predicted_label))
    print("Accuracy:", accuracy_score(y_test, predicted_label))
    print(classification_report(y_test, predicted_label))


    

SVM_model_trainning()
naive_model_trainning()
logistic_model_trainning()
decision_tree_model_trainning()
random_forest_model_trainning()
# conclusion

print("conclusion about the performance of all model:")

print(" with the ratio 0.8/0.2 trainning set and test set, Naive model,  SVM model, logistic model and random forest model have the same accuracy scores, which is 0.642. the decision tree has a bit lower accuracy score which is 0.955. when applying  cross-validaition on 5 folds for some models, we can see that the mean accuracy of validation on SVM is the best closed to accuracy score on test set  in comparison with logistic model and naive model.\n when changing the ratio of trainning set and test set to 0.7/03, the performance of SVM is much more better than 4 other, which is 0.82 in terms of accuracy score.")


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('./creditcard.csv')
count_class = pd.value_counts(data['Class'], sort=True)
count_class.plot(kind='bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

scaler = StandardScaler()
data['normAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
print(data.head())

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

number_records_fraud = len(data[data.Class == 1])
fraud_indicies = np.array(data[data.Class == 1].index)

normal_indicies = data[data.Class == 0].index

random_normal_indicies = np.random.choice(normal_indicies, number_records_fraud, replace=False)
random_normal_indicies = np.array(random_normal_indicies)

under_sample_indicies = np.concatenate([fraud_indicies, random_normal_indicies])
under_sample_data = data.iloc[under_sample_indicies, :]

X_under_sample_data = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_under_sample_data = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

from sklearn.model_selection import train_test_split
# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = \
    train_test_split(X_under_sample_data, y_under_sample_data, test_size=0.3, random_state=0)

print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))

#Recall = TP/(TP+FN)
def print_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)
    c_param_range = [0.01, 0.1, 1, 10, 100]

    result_table = pd.DataFrame(index=range(len(c_param_range)), columns=['c_paramerter', 'Mean recall score'])
    result_table['c_parameter'] = c_param_range
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')
        recall_accs = []
        i = 0
        for train_index, validation_index in fold.split(y_train_data):
            lr = LogisticRegression(penalty='l2', C=c_param)
            lr.fit(x_train_data.iloc[train_index, :], y_train_data.iloc[train_index, :].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[validation_index, :].values)

            recall_acc = recall_score(y_train_data.iloc[validation_index, :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', i, ': recall score = ', recall_acc)
            i += 1
        # The mean value of those recall scores is the metric we want to save and get hold of.
        result_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    result_table['Mean recall score'] = result_table['Mean recall score'].astype("float32")
    print(result_table['Mean recall score'])
    print(type(result_table['Mean recall score']))
    print(result_table['Mean recall score'].idxmax())
    best_c = result_table.iloc[result_table['Mean recall score']].values
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    return best_c

best_c = print_Kfold_scores(X_train_undersample, y_train_undersample)



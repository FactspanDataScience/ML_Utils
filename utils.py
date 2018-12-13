import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def string_clear(comb, cols):
    print(comb.shape)
    for col in cols:
        col_name = col+'_final'
        comb[col_name] = comb[col].apply(lambda x: None if x is None else str(x).split('|')[0])
        comb = comb.drop([col], axis = 1)
    print(comb.shape)
    return comb

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def impute(data, impute_scheme_cat = 'NA', impute_scheme_cont = 'mean'):
    cols  = missing_values_table(data).reset_index()['index']
    for col in cols:
        print "imputing for %s" %col
        if impute_scheme_cat not in ['NA', 'mode']:
            print("Invalid impute scheme chosen for categorical variable")
            return
        if impute_scheme_cont not in ['mean', 'median']:
            print("Invalid impute scheme chosen for categorical variable")
            return
        if data[col].dtype == 'O' and impute_scheme_cat == 'NA':
            data[col] = data[col].fillna('NA')
        elif data[col].dtype == 'O' and impute_scheme_cat == 'mode':
            data[col] = data[col].fillna(data[col].mode()[0])
        elif data[col].dtype != 'O' and impute_scheme_cont == 'mean':
            data[col] = data[col].fillna(data[col].mean())
        elif data[col].dtype != 'O' and impute_scheme_cont == 'median':
            data[col] = data[col].fillna(data[col].median())
    return data

def transform(df):
    ip = df.shape[1]
    cols = df.columns
    for col in cols:
        if df[col].dtype != 'O' and float(len(df[col].unique()))/float(len(df[col])) > 0.05:
            df[col+'_normalize'] = (df[col]-df[col].mean())/df[col].std()
            df[col+'_log'] = np.log(df[col]+1)
            if len(pd.qcut(df[col], 10, labels = False, duplicates = 'drop').unique())<=1:
                df[col+'_bins'] = pd.qcut(df[col], 10, labels = False, duplicates = 'drop')
    op = df.shape[1]
    if ip == op:
        print("Please define transformation function")
    return df

def interact(df):
    ip = df.shape[1]
    # Define the interactions here
    op = df.shape[1]
    if ip == op:
        print("Please define interaction function")
    return df

def encode(df, cols = None, drop_first = True):
    if cols == None:
        return pd.get_dummies(df, cols, drop_first = drop_first)
    else:
        prefix = {cols[i]:cols[i] for i in range(len(cols))}
        return pd.get_dummies(df, prefix, cols, drop_first = drop_first)

def roc(ts_y, preds, classifier):
    fpr, tpr, threshold = roc_curve(ts_y, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic for %s' %classifier)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def classifer(op, md,truth = 'truth'):
    true = op.loc[op[truth]==1, md]
    false = op.loc[op[truth]==0,md]
    plt.hist(x = true, bins = 50,  label = 'Truth' , alpha = 0.5)
    plt.hist(x = false, bins = 50, label = 'False' , alpha = 0.3)
    plt.legend(loc = 'upper right')
    plt.title(md)
    plt.show()

def model(idx, tr_x, tr_y, ts_x, ts_y, feature_imp, estimator):
    op = pd.DataFrame({'id': idx,
                           'truth': ts_y})
    for est in estimator:
        est.fit(tr_x, tr_y)
        if feature_imp:
            try:
                importance = est.feature_importances_
                feat_imp = pd.DataFrame({'importance':est.feature_importances_})    
                feat_imp['feature'] = tr_x.columns
                feat_imp.sort_values(by='importance', ascending=False, inplace=True)
                # feat_imp = feat_imp.iloc[:top_n]
                plt.barh(feat_imp['feature'][:20],feat_imp['importance'][:20])
                plt.title("Feature Importance from "+str(est).split("(")[0])
                plt.show()
            except:
                print(str(est).split("(")[0]+" doesnot output feature importances")
        preds = est.predict_proba(ts_x)[:,1]
        op[str(est).split("(")[0]] = preds
        roc(ts_y, preds, str(est).split("(")[0])
        classifer(op, str(est).split("(")[0],'truth')
        
    print("creating ensemble...")
    probs = 0
    for est in estimator:
        probs += op[str(est).split("(")[0]]/len(estimator)
    op['ensemble'] = probs
    roc(ts_y,probs, 'ensemble')
    classifer(op, 'ensemble', 'truth')

    return op
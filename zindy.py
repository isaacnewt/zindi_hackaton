# step 1: Problem definition
# Step 2: Collection of data points into data frame

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# Step 6: This shows our evaluation metrics for the chosen model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, plot_confusion_matrix

%matplotlib inline
# zindi-user-behaviour-birthday-challenge

# Training dataset
df = pd.read_csv('.\zindi\Train.csv')
X = df.drop('Target', axis=1)
# X_train = features.values
Xtr_train = X.values
ytr_train = df['Target'].values
# y_train = target.values


# Ideal dataset
df_1 = pd.read_csv('.\zindi\Comments.csv')
df_2 = pd.read_csv('.\zindi\CompetitionPartipation.csv')
df_3 = pd.read_csv('.\zindi\Competitions.csv')
df_4 = pd.read_csv('.\zindi\Discussions.csv')
df_5 = pd.read_csv('.\zindi\Submissions.csv')


# df_4a = ((df_1.merge(df_2)).merge(df_3)).merge(df_4)
# df_all = df_4a.merge(df_5)

# Sample Submisssion dataset
df_S = pd.read_csv('.\zindi\SampleSubmission.csv')

# Test dataset
df_test = pd.read_csv('.\zindi\Test.csv')
X = df_test.drop('User_ID', axis=1)
Xt_test = X.values
df_test['target'] = df_S['Target']
yt_test = df_test['target']
ytr = df['Target'].value_counts()
ytr
X.value_counts()
merged = pd.merge(df_1, df_2, how='outer', left_on='UserID', right_on='UserID', sort=True)
merged1 = pd.merge(merged, df_3, how='outer', left_on='CompID', right_on='CompID', sort=True)
merged2 = pd.merge(merged1, df_4, how='outer', left_on='UserID', right_on='UserID', sort=True)
df_all = pd.DataFrame.drop_duplicates(merged2, subset=['UserID'], keep = 'first', inplace = False)
df_asa = pd.merge(df_all, df_5, how='outer', left_on='UserID', right_on='UserID', sort=True)
df_asa = pd.DataFrame.drop_duplicates(df_asa, subset=['UserID'], keep = 'first', inplace = False)
df_asa = df_asa.reset_index()
if df_asa['FeatureA'].any or df_asa['FeatureB'].any or df_asa['FeatureD'].any or df_asa['FeatureE'].any or df_asa['FeatureF'].any or df_asa['FeatureG'].any:
    df_asa['FeatureAll'] = 1
df_all = df_asa.fillna(0)
# df_all = df_all.reset_index(drop=True)
# PublicRank: 1 -11
rank = {'rank 1': 1, 'rank 2':2, 'rank 3': 3, 'rank 4':4, 'rank 5':5, 'rank 6':6, 'rank 7': 7,
       'rank 8': 8, 'rank 9': 9, 'rank 10': 10, 'rank 11':11}
df_all['PublicRank'] = df_all['PublicRank'].map(rank)

count = {'count 1': 1, 'count 2':2, 'count 3': 3, 'count 4':4, 'count 5':5, 'count 6':6, 'count 7': 7,
       'count 8': 8, 'count 9': 9, 'count 10': 10, 'count 11':11}
df_all['Successful Submission Count'] = df_all['Successful Submission Count'].map(count)

# Submission Count : 1 -10
# we drop columns not necessary for our end product

u_dropped_columns = ['UserID', 'CompID_y', 'Country', 'index', 'CompID_x','DiscID', 'SubDate Year', 'FeatureA',
                     'FeatureB', 'FeatureD', 'FeatureE', 'FeatureF', 'FeatureG', 'DiscDate Year', 
                     'CompEndTime Year', 'CompStartTime Year','FeatureAll']
df_all = df_all.drop(columns = u_dropped_columns)
for i in df_all.columns:
    print(f"Unique {i}'s counts: {df_all[i].nunique()}")
    print(f"{df_all[i].unique()}\n")
    # print('{}'.format(df[i].unique()))

print("------------------------------------------------------------")    
    
for i in df.columns:
    print(f"Unique {i}'s counts: {df[i].nunique()}")
    print(f"{df[i].unique()}\n")
    # print('{}'.format(df[i].unique()))
# Columns for Feature rescaling, , 'FeatureF', 'FeatureG'

features_rs = ['FeatureC', 'Points Reward', 'SubmissionLimitPerDay']
df_rs = pd.DataFrame(df_all, columns = features_rs)
df_odas = df_all.drop(columns=features_rs)
# Feature Rescaling

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

rs_feat = mms.fit_transform(df_rs)

df_feat_rs = pd.DataFrame(rs_feat, columns = features_rs, index=df_rs.index)

df_all = pd.concat([df_odas, df_feat_rs], axis=1)

# Let us create a column for target output
df_all['t_output'] = (df['Target'] == 0).astype('int')
# df['t_output'].head(50)
# df.Target.value_counts()
df_all.t_output.value_counts()
# prevalence level means the percentage that will miss out. So below, we have about 14% that will will miss out in being active
def d_prevalence(p):
    return (sum(p)/len(p))

d_prevalence(df_all.t_output.values)
# y_train = df['t_output']
# d_prevalence(ytr_train.values)
# generate a function to generate box plot for major features

plots = {1:[111], 2:[121,122], 3:[131,132,133], 4:[221, 222, 223, 224], 
         5:[231, 232, 233, 234, 235], 6:[231, 232, 233, 234, 235, 236]}

def plotly(x, y, df):
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    
    plt.figure(figsize=(7*columns, 7*rows))
    
    for i,j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.boxplot(x=x, y=j, data=df[[x,j]], linewidth=1, palette="Blues")
        ax.set_title(j)
        
    return plt.show()
plotly("Target", ["month","year","CompPart","Comment","Sub","Disc"], df)
# Generate a function to depict 'plotly' of features
def count_of_plot(x, y, df):
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    
    plt.figure(figsize=(7*columns, 7*rows))
    
    for i,j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.countplot(x=j, hue=x, data=df[[x,j]], linewidth=0.4, edgecolor = 'Black', alpha = 0.8, palette="Blues")
        ax.set_title(j)
        
    return plt.show()
  
count_of_plot("Target", ["month","year","CompPart","Comment","Sub","Disc"], df)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(4, 6))
ax = sns.countplot(x=ytr_train, palette="Blues", linewidth=1)
plt.show


plt.figure(figsize=(16,10))
df.corr()['Target'].sort_values(ascending=False).plot(kind='bar', figsize=(20,5))

df_all['CompStartTime Day_of_week'] = pd.to_datetime(df_all['CompStartTime Day_of_week'], format= '%Y-%m-%dT%H:%M:%SZ', errors = 'coerce')
# Shuffle the sample

df_all = df_all.sample(n=len(df_all), random_state=42)
df_all = df_all.reset_index(drop=True)

#Split the data

df_valid = df_all.sample(frac=0.3, random_state=42)
df_train = df_all.drop(df_valid.index)

# Step 5: Splitting

# from sklearn.model_selection import train_test_split

# y = df_all['t_output'].values
# X = df_all.drop('t_output', axis=1)
# X = X.values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# Feature combination

features = ['CommentDate Year', 'CommentDate Day_of_week', 'CommentDate Month', 'CommentDate Day_of_week', 'PublicRank', 
            'Successful Submission Count', 'CompPartCreated Year', 'CompPartCreated Month', 'CompPartCreated Day_of_week', 
            'Kind', 'SecretCode','CompEndTime Month','CompEndTime Day_of_week','CompStartTime Month', 'CompStartTime Day_of_week', 
            'DiscDate Month','DiscDate Day_of_week','SubDate Month','SubDate Day_of_week','FeatureC',
            'Points Reward','SubmissionLimitPerDay','t_output']

X_train = df_train[features].values
y_train = df_train['t_output'].values

X_test = df_valid[features].values
y_train = df_valid['t_output'].values

#1 categories 

categories = np.union1d(train, test)

train = train.astype('category', categories = categories)
test = test.astype('category', categories = categories)

pd.get_dummies(train)
pd.get_dummies(test)


# We setup a function to display feature weights of classifiers
def we_feat(x_frame, classifier, classifier_name):
    w = pd.Series(classifier.coef[0], index=x_frame.columns.values).sort_values(ascending=False)
    
    topoff = w[:10]
    plt.fig(figsize=(6,4))
    plt.tick_params(label_size=10)
    plt.title(f'{classifier_name} - top 10 weight features')
    topoff.plot(kind='bar')
    
    bottom = w[:,10]
    plt.fig(figsize=(6,4))
    plt.tick_params(label_size=10)
    plt.title(f'{classifier_name} -bottom 10 features')
    bottom.plot(kind='bar')
    
    return print("")

def confuse_plot(y_test, X_test, y_train, X_train, y_pred, classifier, classifier_name):
    # We define plots for the confusion matrix and accuracy score
    
    fig,ax = plt.subplot(7,6)
    plot_confusion_matrix(classifier, X_test, y_test, display_labels=["Non-Active", "Active"], 
                          normalize=None, ax=ax, cmp=plt.cm.blues)
    ax.set_title(f'{classifier_name} - Confussion matix plot')
    plt.show()
    
    fig,ax = plt.subplot(7,6)
    plot_confusion_matrix(classifier, X_test, y_test, display_labels=["Non-Active", "Active"], 
                          normalize=True, ax=ax, cmp=plt.cm.blues)
    ax.set_title(f'{classifier_name} - Confussion matix plot(norm)')
    plt.show()
    
    print(f'Accuracy score test:{accuracy_score(y_test, y_pred)}')
    print(f'Accuracy score train: {classifier.score(y_train, X_train)} - (Comparing the training performance)' )
    return print(" ")

def rocurve_auscore(X_test, y_test, classifier, y_ped_probabilities):
    # We now display plots for roc, auc, recall metrics
    
    y_pred_prob = y_pred_probabilities[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr, tpr, f'{classifier_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier_name} - ROC')
    plt.show()
    return print(f'AUC Score(ROC): {roc_auc_score(y_test, y_pred_prob)}\n')

def scores_prcurve(classifier_name, y_test, y_pred, y_pred_probabilities):
    # we will further display the metrics for preci_recall_curve and other scores
    
    y_pred_prob = y_pred_probabilities[:,1]
    precision, recall, thresholds =  precision_recall_curve(y_test, y_pred_prob)
    
    plt.plot(recall, precision, "Precision Recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{classifier_name} - PR Curve')
    pl.show()
    
    f1, auc_score = f1_score(y_test, y_pred), auc(recall, precision)
    return print(f'F1_score_result: {f1} \n AUC Score (ROC): {auc_score}\n')
    
# Step 7: Model selection, Training, Prediction and Assessment
from sklearn.ensemble import RandomForestClassifier

# Instantiate, train the model, make predictions and evaluate chosen metric
def model(X_test , y_test, y_train):
    # Instantiating the classifier...
    rf=RandomForestClassifier(max_depth = 5, n_estimators=100, random_state = 42)
    rf.fit(X_train, y_train)
    
    # Making class prediction with associated probability
    y_pred = rf.predict(X_test)
    y_pred_prob = rf.predict_proba(X_test)
    
    # plot model evaluations
    scores_prcurve(classifier_name = rf().__class__.__name__, y_test =y_t, y_pred=y_pred, y_pred_prob=y_pred_prob)
    rocurve_auscore(X_t, y_t, classifier = rf().__class__.__name__, y_pred_prob =y_ped_prob)
    confuse_plot(y_t, X_t, y_train, X_train, y_pred, classifier = rf, classifier_name = rf().__class__.__name__)
    


  

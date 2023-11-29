#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler


# # Exploratory Data Analysis

# In[2]:


data = pd.read_csv("loan_data_2007_2014.csv")


# In[3]:


# Menampilkan semua kolom dan baris
pd.set_option("display.max_columns", None)

# Menampilkan DataFrame
data


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


kolom_to_drop = ['Unnamed: 0']
data = data.drop(kolom_to_drop, axis=1)


# In[7]:


data.head()


# In[8]:


import pandas as pd

# 1. Apakah format data sudah teragregasi atau data masih raw?
print("1. Format Data: Data masih dalam format raw atau mentah.")

# 2. Satu baris data mewakili apa?
print("2. Satu Baris Data Mewakili: Satu baris data mewakili profil seorang peminjam atau pengguna layanan pinjaman.")

# 3. Apakah ada satu individu sama yang menempati lebih dari satu baris (duplicate)?
is_duplicate = data.duplicated(subset=['id']).any()
print(f"3. Duplikasi Data: {is_duplicate}")

# 4. Kolom-kolom apa saja yang tersedia?
print("4. Kolom-Kolom yang Tersedia:")
print(data.columns)

# 5. Apakah data yang tersedia cukup untuk membuat pemodelan credit risk?
print("Data ini memiliki banyak informasi yang relevan untuk pemodelan risiko kredit, seperti jumlah pinjaman, tingkat suku bunga, pendapatan tahunan, status pekerjaan, dan lain-lain. Namun, untuk membuat pemodelan risiko kredit yang baik, Anda mungkin perlu melakukan pra-pemrosesan data, seperti mengatasi missing values, mengidentifikasi dan mengatasi outlier, serta memilih fitur-fitur yang paling relevan. Selain itu, Anda juga perlu melakukan analisis lebih lanjut terkait dengan distribusi variabel dan korelasi antar variabel sebelum membangun model.")


# In[9]:


# Dapatkan daftar jenis semua kolom di dataset
data.dtypes


# In[10]:


data.info()


# In[11]:


data.columns


# In[12]:


data.isnull().sum()


# In[13]:


# Melihat nilai unik dalam setiap kolom
for column in data.columns:
    unique_values = data[column].unique()
    print(f"Kolom '{column}': {unique_values}")


# # Defining Label

# In[14]:


column = 'loan_status'
unique_values = data[column].value_counts()
print(f"Kolom '{column}':")
print(unique_values.index.tolist())


# In[15]:


# Mengubah nilai 'Fully Paid' dan 'Current' menjadi 1, yang lainnya menjadi 0
data['loan_status'] = data['loan_status'].apply(lambda x: 1 if x in ['Fully Paid', 'Current'] else 0)

data.head()


# # Feature Engineering

# In[16]:


data.info()


# In[17]:


# Melihat nilai unik dalam setiap kolom
for column in data.columns:
    unique_values = data[column].unique()
    print(f"Kolom '{column}': {unique_values}")


# In[18]:


# Menampilkan nilai unik dari setiap kolom
print("Nilai unik dari kolom 'issue_d':")
print(data['issue_d'].unique())

print("\nNilai unik dari kolom 'earliest_cr_line':")
print(data['earliest_cr_line'].unique())

print("\nNilai unik dari kolom 'last_pymnt_d':")
print(data['last_pymnt_d'].unique())

print("\nNilai unik dari kolom 'next_pymnt_d':")
print(data['next_pymnt_d'].unique())

print("\nNilai unik dari kolom 'last_credit_pull_d':")
print(data['last_credit_pull_d'].unique())


# In[19]:


# Mengubah format kolom 'issue_d'
data['issue_d'] = data['issue_d'].str[-2:]

# Mengubah format kolom 'earliest_cr_line'
data['earliest_cr_line'] = data['earliest_cr_line'].str[-2:]

# Mengubah format kolom 'last_pymnt_d'
data['last_pymnt_d'] = data['last_pymnt_d'].str[-2:]

# Mengubah format kolom 'next_pymnt_d'
data['next_pymnt_d'] = data['next_pymnt_d'].str[-2:]

# Mengubah format kolom 'last_credit_pull_d'
data['last_credit_pull_d'] = data['last_credit_pull_d'].str[-2:]


# In[20]:


data.head()


# In[21]:


# Contoh: Menghitung rata-rata 'loan_amnt' berdasarkan 'purpose'
average_loan_by_purpose = data.groupby('purpose')['loan_amnt'].mean()

# Contoh: Menghitung jumlah total 'annual_inc' berdasarkan 'home_ownership'
total_annual_income_by_ownership = data.groupby('home_ownership')['annual_inc'].sum()

# Contoh: Menghitung median 'dti' berdasarkan 'grade'
median_dti_by_grade = data.groupby('grade')['dti'].median()

mean_annual_inc_by_grade = data.groupby('grade')['annual_inc'].mean()

# Menampilkan hasil agregasi
print("Rata-rata loan_amnt berdasarkan purpose:")
print(average_loan_by_purpose)

print("\nTotal annual_inc berdasarkan home_ownership:")
print(total_annual_income_by_ownership)

print("\nMedian dti berdasarkan grade:")
print(median_dti_by_grade)

print("\nMean annual_inc berdasarkan grade:")
print(mean_annual_inc_by_grade)


# In[22]:


# Agregasi rata-rata loan_amnt berdasarkan purpose
average_loan_by_purpose = data.groupby('purpose')['loan_amnt'].mean().reset_index()
average_loan_by_purpose.columns = ['purpose', 'avg_loan_amnt']

# Agregasi total annual_inc berdasarkan home_ownership
total_annual_income_by_ownership = data.groupby('home_ownership')['annual_inc'].sum().reset_index()
total_annual_income_by_ownership.columns = ['home_ownership', 'total_annual_inc']

# Agregasi median dti berdasarkan grade
median_dti_by_grade = data.groupby('grade')['dti'].median().reset_index()
median_dti_by_grade.columns = ['grade', 'median_dti']

# Agregasi mean annual_inc berdasarkan grade
mean_annual_inc_by_grade = data.groupby('grade')['annual_inc'].mean().reset_index()
mean_annual_inc_by_grade.columns = ['grade', 'mean_annual_inc']

# Gabungkan hasil agregasi dengan data asli
data = data.merge(average_loan_by_purpose, on='purpose', how='left')
data = data.merge(total_annual_income_by_ownership, on='home_ownership', how='left')
data = data.merge(median_dti_by_grade, on='grade', how='left')
data = data.merge(mean_annual_inc_by_grade, on='grade', how='left')


# In[23]:


data.head()


# # Feature Selection

# In[24]:


columns_to_drop = [
    'id', 'member_id', 'url', 'desc',
    'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
    'annual_inc_joint', 'dti_joint', 'verification_status_joint',
    'emp_title', 'title', 'zip_code',
    'pymnt_plan', 'policy_code',
    'application_type', 'initial_list_status', 'term', 'open_acc_6m',
    'open_il_6m', 'open_il_12m', 'open_il_24m',	'mths_since_rcnt_il', 'total_bal_il',
    'il_util',	'open_rv_12m', 'open_rv_24m', 'max_bal_bc',	'all_util', 'inq_fi', 'total_cu_tl',
    'inq_last_12m', 'next_pymnt_d', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'
]

data = data.drop(columns=columns_to_drop)


# In[25]:


data.head()


# In[26]:


data.info()


# In[27]:


# Melihat nilai unik dalam setiap kolom
for column in data.columns:
    unique_values = data[column].unique()
    print(f"Kolom '{column}': {unique_values}")


# # Handling Missing Values

# In[28]:


data.isnull().sum()


# In[29]:


data = data.dropna()


# In[30]:


# Melihat nilai unik dalam setiap kolom
for column in data.columns:
    unique_values = data[column].unique()
    print(f"Kolom '{column}': {unique_values}")


# In[31]:


data.info()


# # Feature Scaling dan Encoding

# In[32]:


data['issue_d'] = pd.to_numeric(data['issue_d'])
data['earliest_cr_line'] = pd.to_numeric(data['earliest_cr_line'])
data['last_pymnt_d'] = pd.to_numeric(data['last_pymnt_d'])
data['last_credit_pull_d'] = pd.to_numeric(data['last_credit_pull_d'])


# In[33]:


data.info()


# In[34]:


# Melihat nilai unik dalam setiap kolom
for column in data.columns:
    unique_values = data[column].unique()
    print(f"Kolom '{column}': {unique_values}")


# In[35]:


data.head()


# In[36]:


data.columns


# In[37]:


# Kolom yang akan di-encode
columns_to_encode = ['grade', 'sub_grade', 'emp_length', 'home_ownership', 
                     'verification_status', 'purpose', 'addr_state']

# Inisialisasi dan fitting OrdinalEncoder
encoder = OrdinalEncoder()
data[columns_to_encode] = encoder.fit_transform(data[columns_to_encode])


# In[38]:


data


# # Modeling - Train

# In[39]:


target_name = 'loan_status'
labels_dataset = data[target_name]
features_dataset = data.drop(target_name, axis=1)


# In[40]:


features_dataset


# In[41]:


labels_dataset


# In[42]:


features_dataset = MinMaxScaler().fit_transform(features_dataset)


# In[43]:


random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    features_dataset, labels_dataset, train_size=0.8, random_state=random_seed
)


# In[44]:


rfc = RandomForestClassifier()
rfc.fit(train_features, train_labels)


# In[45]:


train_score = rfc.score(train_features, train_labels)
test_score = rfc.score(test_features, test_labels)

print(f"Support Vector Classifier on the training dataset: {train_score:.2f}")
print(f"Support Vector Classifier on the test dataset:     {test_score:.2f}")


# # Modeling - Evaluation

# In[46]:


rfc_pred = rfc.predict(test_features)


# In[47]:


cm_rfc = confusion_matrix(test_labels,rfc_pred)
cm_rfc


# In[48]:


sns.heatmap(confusion_matrix(test_labels,rfc_pred),annot=True,fmt="d")


# In[49]:


TN = cm_rfc[0,0]
FP = cm_rfc[0,1]
FN = cm_rfc[1,0]
TP = cm_rfc[1,1]


# In[50]:


TN, FP, FN, TP


# In[51]:


cm_rfc = confusion_matrix(test_labels, rfc_pred)

print('TN - True Negative {}'.format(cm_rfc[0,0])) 
print('FP - False Positive {}'.format(cm_rfc[0,1]))
print('FN - False Negative {}'.format(cm_rfc[1,0]))
print('TP - True Positive {}'.format(cm_rfc[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm_rfc[0,0],cm_rfc[1,1]]),np.sum(cm_rfc))*100))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm_rfc[0,1],cm_rfc[1,0]]),np.sum(cm_rfc))*100))


# In[52]:


plt.clf()
plt.imshow(cm_rfc, interpolation='nearest', cmap=plt.cm.Wistia) 
classNames = ['0','1']
plt.title('Confusion Matrix Of Random Forest Classifier')
plt.ylabel('Actual(true) values')
plt.xlabel('Predicted values')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN' , 'FP'], ['FN' , 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm_rfc[i][j]))


# In[53]:


pd.crosstab(test_labels, rfc_pred, margins=False)


# In[54]:


pd.crosstab(test_labels, rfc_pred, margins=True)


# In[55]:


pd.crosstab(test_labels, rfc_pred, rownames=['Actual values'], colnames=['Predicted values'], margins=True)


# In[56]:


TP,FP


# In[57]:


Precision = TP/(TP+FP)
Precision


# In[58]:


# print precision score

precision_Score_rfc = TP / float(TP + FP)*100
print('Precision score: {0:0.4f}'.format(precision_Score_rfc))


# In[59]:


print("precision Score is:", precision_score (test_labels, rfc_pred)*100)
print("Mircro Average precision Score is:", precision_score(test_labels, rfc_pred, average='micro')*100)
print("Marcro Average precision Score is:", precision_score (test_labels, rfc_pred, average='macro')*100)
print("Weighted Average precision Score is:", precision_score (test_labels, rfc_pred, average='weighted')*100)
print("precision Score on Non weighted score is:", precision_score(test_labels, rfc_pred, average=None)*100)


# In[60]:


print('Classification Report of Random Forest Classifier: \n',classification_report(test_labels, rfc_pred, digits=4))


# In[61]:


recall_score_rfc = TP / float(TP + FN)*100
print('recall score', recall_score_rfc)


# In[62]:


TP,FN


# In[63]:


print('Recall or Sensitivity score :',recall_score(test_labels,rfc_pred)*100)


# In[64]:


print("Mircro Average Recall Score is", recall_score(test_labels, rfc_pred, average='micro')*100)
print("Marcro Average Recall Score is", recall_score(test_labels, rfc_pred, average='macro')*100)
print("Weighted Average Recall Score is:", recall_score(test_labels, rfc_pred, average='weighted')*100)
print("Recall Score on Non weighted score is:", recall_score(test_labels, rfc_pred, average=None)*100)


# In[65]:


print('Classification Report of Random Forest Classifier: \n',classification_report(test_labels,rfc_pred,digits=4))


# In[66]:


FPR_rfc = FP / float(FP + TN)*100
print('False Positive Rate: {0:0.4f}'.format(FPR_rfc))


# In[67]:


FP, TN


# In[68]:


specificity_rfc = TN / (TN + FP)*100
print('Specificity: {0:0.4f}'.format(specificity_rfc))


# In[69]:


f1_score_rfc = f1_score(test_labels, rfc_pred)*100
print("f1_score of macro :",f1_score_rfc)


# In[70]:


print("Mircro Average F1 Score is:", f1_score(test_labels, rfc_pred, average='micro')*100)
print("Marcro Average F1 Score is:", f1_score(test_labels, rfc_pred, average='macro')*100)
print("Weighted Average F1 Score is:", f1_score(test_labels, rfc_pred, average='weighted')*100)
print("F1 Score on Non weighted score is:", f1_score(test_labels, rfc_pred, average=None)*100)


# In[71]:


print('Classification Report of Random Forest Classifier: \n', classification_report(test_labels,rfc_pred,digits=4))


# In[72]:


# Area Under Curve 
auc_rfc = roc_auc_score(test_labels, rfc_pred)
print("ROC AUC SCORE of Artificial Neural Network (ANN) is", auc_rfc)


# In[73]:


fpr, tpr, thresholds = roc_curve(test_labels, rfc_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area - %0.2f)' % auc_rfc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of Random Forest Classifier')
plt.legend()
plt.grid()
plt.show()


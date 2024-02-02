#!/usr/bin/env python
# coding: utf-8

# #### Import Modules

# In[3]:


import pandas as pd # olah dan analisis data


# #### Load dataset

# In[4]:


cc_df = pd.read_csv('https://raw.githubusercontent.com/spaceforglory/OPENSOURCEUAS/main/covid_19_clean_complete.csv') # memuat file csv sebagai data frame
cc_df.head() # tampilkan 5 baris pertama


# #### Drop column 'Id'

# In[5]:


cc_df.drop(columns='Province/State', inplace=True) # menghapus kolom bernama 'Province/State'
cc_df.head() # tampilkan 5 baris pertama


# #### Identify the shape of the datatset

# In[6]:


cc_df.shape # bentuk/dimensi dataset (baris,kolom)


# #### Get the list of columns

# In[7]:


cc_df.columns # daftar nama kolom


# #### Identify data types for each column

# In[8]:


cc_df.dtypes # tipe data untuk tiap kolom


# #### Get bassic dataset information

# In[9]:


cc_df.info() # informasi dataset


# #### Identify missing values

# In[10]:


cc_df.isna().values.any() # mendeteksi keberadaan nilai kosong


# #### Identify duplicate entries/rows

# In[12]:


# tampilkan seluruh baris dengan duplikasi
cc_df[cc_df.duplicated()] # tampilkan hanya baris duplikasi sekunder


# In[13]:


cc_df.duplicated().value_counts() # hitung jumlah duplikasi data


# #### Drop duplicate entries/rows

# In[14]:


cc_df.drop_duplicates(inplace=True) # menghapus duplikasi data
cc_df.shape


# #### Describe the dataset

# In[15]:


cc_df.describe() # deskripsi data


# #### Correlation Matrix

# In[20]:


cc_df.drop(columns='Country/Region', inplace=True) # menghapus kolom 
cc_df.head() # tampilkan 5 baris pertama


# In[21]:


cc_df.drop(columns='Lat', inplace=True) # menghapus kolom 
cc_df.head() # tampilkan 5 baris pertama, 'Long', 'Date', 'WHO Region',


# In[22]:


cc_df.drop(columns='Long', inplace=True) # menghapus kolom 
cc_df.head() # tampilkan 5 baris pertama, 'Long', 'Date', 'WHO Region',


# In[23]:


cc_df.drop(columns='Date', inplace=True) # menghapus kolom 
cc_df.head() # tampilkan 5 baris pertama, 'Long', 'Date', 'WHO Region',


# In[24]:


cc_df.drop(columns='WHO Region', inplace=True) # menghapus kolom 
cc_df.head() # tampilkan 5 baris pertama, 'Long', 'Date', 'WHO Region',


# In[25]:


cc_df.corr() # korelasi antar kolom


# ## Iris Dataset: Data Visualisation

# #### Import Modules

# In[26]:


import matplotlib.pyplot as plt # visualisasi data
import seaborn as sns # visualisasi data

# output dari visualisasi data akan diarahkan ke notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Heatmap

# In[27]:


sns.heatmap(data=cc_df.corr())


# #### Bar Plot

# In[28]:


cc_df['Deaths'].value_counts() # menghitung jumlah


# In[30]:


cc_df['Deaths'].value_counts().plot.bar()
plt.tight_layout()
plt.show()


# In[33]:


sns.countplot(data=cc_df, x='Deaths')
plt.tight_layout()
# sns.countplot?


# #### Pie Chart

# In[35]:


cc_df['Deaths'].value_counts().plot.pie(autopct='%1.1f%%', labels=None, legend=True)
plt.tight_layout()


# #### Line Plot

# In[36]:


fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

cc_df['Deaths'].plot.line(ax=ax[0][0])
ax[0][0].set_title('Deaths')

cc_df['Recovered'].plot.line(ax=ax[0][1])
ax[0][1].set_title('Recovered')

cc_df.PetalLengthCm.plot.line(ax=ax[1][0])
ax[1][0].set_title('Petal Length')

cc_df.PetalWidthCm.plot.line(ax=ax[1][1])
ax[1][1].set_title('Petal Width')


# In[37]:


cc_df.plot()
plt.tight_layout()


# #### Histogram

# In[38]:


cc_df.hist(figsize=(6,6), bins=10)
plt.tight_layout()


# #### Boxplot

# In[39]:


cc_df.boxplot()
plt.tight_layout()


# In[ ]:


cc_df.boxplot(by="Deaths", figsize=(8,8))
plt.tight_layout()


# #### Scatter Plot

# In[ ]:


sns.scatterplot(x='Deaths', y='Recovered', data=cc_df, hue='Confirmed')
plt.tight_layout()


# #### Pair Plot

# In[ ]:


sns.pairplot(cc_df, hue='Confirmed', markers='+')
plt.tight_layout()


# #### Violin Plot

# In[ ]:


sns.violinplot(data=cc_df, y='Confirmed', x='Active', inner='Deaths')
plt.tight_layout()


# #### Import Modules

# In[ ]:


from sklearn.model_selection import train_test_split # pembagi dataset menjadi training dan testing set
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report # evaluasi performa model


# #### Dataset: Features & Class Label

# In[ ]:


X = cc_df.drop(columns='Deaths') # menempatkan features ke dalam variable X
X.head() # tampilkan 5 baris pertama


# In[ ]:


y = cc_df['Deaths'] # menempatkan class label (target) ke dalam variabel y
y.head() # tampilkan 5 baris pertama


# #### Split the dataset into a training set and a testing set

# In[ ]:


# membagi dataset ke dalam training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

print('training dataset')
print(X_train.shape)
print(y_train.shape)
print()
print('testing dataset:')
print(X_test.shape)
print(y_test.shape)


# #### K Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


k_range = list(range(1,26))
scores = []
for k in k_range:
    model_knn = KNeighborsClassifier(n_neighbors=k) # konfigurasi algoritma
    model_knn.fit(X_train, y_train) # training model/classifier
    y_pred = model_knn.predict(X_test) # melakukan prediksi
    scores.append(accuracy_score(y_test, y_pred)) # evaluasi performa


# In[ ]:


plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.tight_layout()
plt.show()


# In[ ]:


model_knn = KNeighborsClassifier(n_neighbors=3) # konfigurasi algoritma
model_knn.fit(X_train,y_train) # training model/classifier
y_pred = model_knn.predict(X_test) # melakukan prediksi


# ##### Accuracy Score

# In[ ]:


print(accuracy_score(y_test, y_pred)) # evaluasi akurasi


# ##### Confusion Matrix

# In[ ]:


print(confusion_matrix(y_test, y_pred)) # evaluasi confusion matrix


# ##### Classification Report

# In[ ]:


print(classification_report(y_test, y_pred)) # evaluasi klasifikasi


# #### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


# model_logreg = LogisticRegression()
model_logreg = LogisticRegression(solver='lbfgs', multi_class='auto')
model_logreg.fit(X_train,y_train)
y_pred = model_logreg.predict(X_test)


# ##### Accuracy Score

# In[ ]:


print(accuracy_score(y_test, y_pred))


# ##### Confusion Matrix

# In[ ]:


print(confusion_matrix(y_test, y_pred))


# ##### Classification Report

# In[ ]:


print(classification_report(y_test, y_pred))


# #### Support Vector Classifier

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


# model_svc = SVC()
model_svc = SVC(gamma='scale')
model_svc.fit(X_train,y_train)
y_pred = model_svc.predict(X_test)


# #### Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
y_pred = model_dt.predict(X_test)


# #### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# model_rf = RandomForestClassifier()
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train,y_train)
pred_rf = model_rf.predict(X_test)


# #### Accuracy comparision for various models.

# In[ ]:


models = [model_knn, model_logreg, model_svc, model_dt, model_rf]
accuracy_scores = []
for model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
print(accuracy_scores)


# In[ ]:


plt.bar(['KNN', 'LogReg', 'SVC', 'DT', 'RF'],accuracy_scores)
plt.ylim(0.90,1.01)
plt.title('Accuracy comparision for various models', fontsize=15, color='r')
plt.xlabel('Models', fontsize=18, color='g')
plt.ylabel('Accuracy Score', fontsize=18, color='g')
plt.tight_layout()
plt.show()


# In[ ]:





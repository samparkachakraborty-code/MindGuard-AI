import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


#task - 1 load the dataset : 

df = pd.read_csv(" ")    #download data from official kaggel dataset

print(df.info())    # for the basic overview of the data , for my case , here is no missing values .

#task - 2 , use getdummies for columns such as : gender , 

df_encoded = pd.get_dummies(df , columns=['gender' ,'platform_usage' , 'social_interaction_level' ] , drop_first=True)
df_encoded = df_encoded.astype(float)          #so that the math doesn't effect .
print(df_encoded)

#task-3 , now features X and the target y : 

X = df_encoded.drop(['depression_label', 'stress_level' , 'anxiety_level' , 'addiction_level' ] , axis=1)
y = df_encoded['depression_label']

#task-4 , now train , test , split the data : 

X_train ,X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

# task -5 , scale the data : 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# task- 6 , now initialize the smote : 

smote = SMOTE(random_state=42)

X_train_res  , y_train_res= smote.fit_resample(X_train_scaled , y_train)

# task- 7 , initialize the model : 

model = RandomForestClassifier(n_estimators=100 , random_state=42 , class_weight='balanced')

#task-8 , fit the model & prediction made by the model  : 

model.fit(X_train_res , y_train_res)

prediction = model.predict(X_test_scaled)

# [We got 0 in the class 1 , so I'm about to use the probability here : ]

y_prob = model.predict_proba(X_test_scaled)[: , 1]
threshold = 0.2
prediction = (y_prob >= threshold).astype(int)


#task-9 , classification report : 

print("------- Classification Report --------")

print(classification_report(y_test , prediction)) 
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index, palette='magma')
plt.title('Impact Factors on Teen Depression')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


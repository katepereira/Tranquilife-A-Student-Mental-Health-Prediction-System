#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
# Reading the dataset
data = pd.read_csv("Student_Mental_Health.csv")

#drop age and cgpa column
age_column = data['Age']
data = data.drop(columns=['Age'])
CGPA = data['What is your CGPA?']
data = data.drop(columns=['What is your CGPA?'])

new_data = pd.get_dummies(data, columns=["Choose your gender","Marital status", "Do you have Depression?", "Do you have Anxiety?", "Do you have Panic attack?", "Did you seek any specialist for a treatment?","Do you need Help?"])

new_data["Choose your gender"] = new_data["Choose your gender_Female"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Choose your gender_Female", "Choose your gender_Male"], inplace=True)


# Combine "Marital status_Yes" and "Marital status_No" into a single binary column
new_data["Marital status"] = new_data["Marital status_Yes"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Marital status_Yes", "Marital status_No"], inplace=True)

new_data["Do you have Depression?"] = new_data["Do you have Depression?_Yes"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Do you have Depression?_Yes", "Do you have Depression?_No"], inplace=True)

new_data["Do you have Anxiety?"] = new_data["Do you have Anxiety?_Yes"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Do you have Anxiety?_Yes", "Do you have Anxiety?_No"], inplace=True)

new_data["Do you have Panic attack?"] = new_data["Do you have Panic attack?_Yes"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Do you have Panic attack?_Yes", "Do you have Panic attack?_No"], inplace=True)
new_data["Did you seek any specialist for a treatment?"] = new_data["Did you seek any specialist for a treatment?_Yes"]

# Drop the original "Marital status" columns
new_data.drop(columns=["Did you seek any specialist for a treatment?_Yes", "Did you seek any specialist for a treatment?_No"], inplace=True)

new_data["Do you need Help?"] = new_data["Do you need Help?_1"]
new_data.drop(columns=["Do you need Help?_1", "Do you need Help?_0"], inplace=True)


#adding cgpa and age columns again
data_final = new_data.copy()
data_final.insert(3, 'Age', age_column)
data_final.insert(4, 'CGPA', CGPA)
# Converting it to an array
data_final = np.array(data_final)
# Data Slicing
X = data_final[0:, 2:-1]
y = data_final[0:, -1]
X = X.astype('float')
y = y.astype('float')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

# inputt=[int(x) for x in "45 32 60".split(' ')]
# final=[np.array(inputt)]

# b = log_reg.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))



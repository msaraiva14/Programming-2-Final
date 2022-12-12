#### Import Packages 
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import altair as alt

#### App Visuals
st.title("Do you use LinkedIn?")

#Define Income

income = st.selectbox("Income Level",
                    options = ["Less than 10,000",
                               "10,000 to under 20,000",
                               "20,000 to under 30,000",
                               "30,000 to under 40,000",
                               "40,000 to under 50,000",
                               "50,000 to under 60,000",
                               "60,000 to under 70,000",
                               "75,000 to under 100,000",
                               "100,000 to under 150,000",
                               "150,000 or more"])

st.write(f"Income selected:{income}")

#st write("Convert Selection to Numeric Values")

if income == "Less than 10,000":
    income = 1
elif income == "10,000 to under 20,000":
    income = 2
elif income == "20,000 to under 30,000":
    income = 3
elif income == "30,000 to under 40,000":
    income = 4
elif income == "40,000 to under 50,000":
    income = 5
elif income == "50,000 to under 75,000":
    income = 6
elif income == "75,000 to under 100,000":
    income = 7
elif income == "100,000 to under 150,000":
    income = 8
else: 
    income = 9


#Education 
education = st.selectbox("Degree",
                    options = ["Less than High School",
                               "High School, Incomplete",
                               "High School, Graduate",
                               "Some College, No Degree",
                               "Two Year Associate Degree",
                               "Four Year Bachelor's Degree",
                               "Some Postgraduate, Postgraduate or Professional"])

st.write(f"Degree selected: {education}")

#Convert selection to numeric
if education == "Less than High School":
    education = 1
elif education == "High School, Incomplete":
    education = 2
elif education == "High School, Graduate":
    education = 3
elif education == "Some College, No Degree":
    education = 4
elif education == "Two Year Associate Degree":
    education = 5 
elif education == "Four Year Bachelor's Degree": 
    education = 6 
elif education == "Some Postgraduate, No Degree":
    education = 7
else: 
    education = 8

#PARENT
parent = st.selectbox("Do you have children?", ('Yes', 'No'))
st.write(f"Parent:{parent}")

if parent == "Yes":
    parent = 1
else: 
    parent = 0

#Married
married = st.selectbox("Are you married?", ('Yes', 'No'))
st.write(f"Married:{married}")

if married == "Yes":
    married = 1
else: 
    married = 0


#Female
female = st.selectbox("What gender do you identify as?", ('Male', 'Female'))
st.write(f"Gender:{female}")

if female == "Female":
    female = 1
else: 
    female = 0

#Age 
age = st.number_input("Enter your Age",
                       min_value = 1, 
                       max_value = 99,
                       value = 30)
                    
st.write("Your age: ",age)  


####Python Code
#read in the data then create dataframe without na
s = pd.read_csv("social_media_usage.csv")
ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, 0),
    "income":np.where(s["income"]> 9,np.nan,s["income"]),
    "education":np.where(s["educ2"]> 8,np.nan,s["educ2"]),
    "parent":np.where(s["par"] == 1,1,0),
    "married": np.where(s["marital"] ==1,1,0),
    "female": np.where(s["gender"] ==2,1,0),
    "age":np.where(s["age"] >98, np.nan,s["age"])})
ss = ss.dropna()

#creating target vector and feature
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

#Test and train data 
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=123) # set for reproducibility

#instantiate logistic regression and fit model to traiing data
lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train)

#evaluate test data 
y_pred = lr.predict(X_test)

#print classification report
print(classification_report(y_test, y_pred))

#New Data LinkedIN Input
New_Adds = pd.DataFrame({
    "income": [income],
    "education": [education],
    "parent": [parent],
    "married": [married],
    "female": [female],
    "age": [age]    
})

#Add if LinkedIn user or not
if age <= 16:
    age_label = "we think you are not a LinkedIn User"
elif age > 16 and age < 53:
    age_label = "we think you are a LinkedIn User"
else:
    age_label = "we think you are not a LinkedIn User"


user = (lr.predict_proba(New_Adds))[0][1]
user = round(user,2)

st.write(f"According to the inputs above {age_label}")
st.write(f"Based on what you have entered, we think the probability that you are a LinkedIn user is {user}.")

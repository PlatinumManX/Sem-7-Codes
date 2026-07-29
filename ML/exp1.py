import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 1. Load dataset
df=pd.read_csv('train.csv')


# Selecting only the required features among these
try:
    new_df=df.drop(columns=['Name','Ticket','PassengerId','Cabin'])
except Exception as e:
    print(e)


# 2. Dataset exploring

# shape of dataset
print("Shape:",new_df.shape)

# dataset information
print("\nDataset info:")
print(new_df.info())

# Dataset describe
print("\nDataset describe:")
print(new_df)

# dataset top 5 value 
print("\nDataset top 5 values:")
print(new_df.head(5))



# 3. Handling missing values

# Missing value counts 
print("\nMissing value counts:")
print(new_df.isnull().sum())

# filling using ML imputer with mean value

imputer=SimpleImputer(strategy="mean")
new_df['Age']=imputer.fit_transform(new_df[['Age']])

# filling in the categorical
new_df['Embarked']=new_df['Embarked'].fillna(new_df['Embarked'].mode()[0])

print("\nMissing value counts:")
print(new_df.isnull().sum())

# 4. Encoding 

# LabelEncoder
le=LabelEncoder()
new_df['Sex']=le.fit_transform(new_df['Sex'])

# One-Hot Encoding
new_df=pd.get_dummies(new_df,columns=["Embarked"],drop_first=True)

print("\nDataset top 5 values:")
print(new_df.head(5))


# 5. Outlier detection
Q1=new_df['Fare'].quantile(0.25)
Q3=new_df['Fare'].quantile(0.75)

IQR=Q3 - Q1

lower= Q1 - 1.5 * IQR
upper= Q3 + 1.5 * IQR

outliers=new_df[(new_df['Fare']<lower) | (new_df['Fare']>upper)]

print("\n Outliers:",outliers)


# 6. Imbalance data balancing
print(new_df['Survived'].value_counts())
smote=SMOTE()
X = new_df.drop("Survived", axis=1)
y = new_df["Survived"]

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("Before:")
print(y.value_counts())

print("\nAfter:")
print(y_balanced.value_counts())


# 7. Data Transformation
scaler = StandardScaler()

new_df[["Age","Fare"]] = scaler.fit_transform(df[["Age","Fare"]])

print("\nDataset top 5 values:")
print(new_df.head(5))

# 8.Split Dataset

# 80-20 splitting
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nFor 80-20 Splitting:")
print("Training input:",X_train.shape)
print("Training output",y_train.shape)
print("Testing input:",X_test.shape)
print("testing output",y_test.shape)

# 70-30
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
print("\nFor 70-30 Splitting:")
print("Training input:",X_train.shape)
print("Training output",y_train.shape)
print("Testing input:",X_test.shape)
print("testing output",y_test.shape)




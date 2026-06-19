import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
assert 'df' in locals(), "Error: You didn't name your dataframe 'df'."
assert len(df) == 891, f"Error: Expected 891 rows, but got {len(df)}. Check your loading code."
print("✅ Data loaded perfectly! Move to the next step.")
df.head()
print(df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].median())
assert df['Age'].isnull().sum() == 0, "Error: There are still missing values in the Age column!"
print("✅ Missing ages handled perfectly!")
print("Missing values handled!")

assert df['Age'].isnull().sum() == 0, "Error: There are still missing values in the Age column!"
print("✅ Missing ages handled perfectly!")


plt.figure(figsize=(5,4))
df['Survived'].value_counts().plot(kind='bar')
plt.title ('survivalrate')
plt.xlabel('Surviving(0= No, 1= Yes)')
plt.ylabel('No of humans')
plt.show()


plt.figure(figsize=(5,4))
df.groupby('Sex')['Survived'].mean().plot(kind='bar')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()


plt.figure(figsize=(6,4)) 
df['Age'].hist(bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

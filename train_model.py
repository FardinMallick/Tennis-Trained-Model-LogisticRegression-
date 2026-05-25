import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score
df = pd.read_csv("D:\\dataset\\play_tennis.csv")
df
print(df.describe())
for column in df.columns:
    print(column)
    print(df[column].unique())
    print()
import matplotlib.pyplot as plt

df['play'].value_counts().plot(kind='bar')
plt.title("Play Tennis Count")
plt.show()
df['play'].value_counts().plot(
    kind='bar',
    color=['red','green']
)
plt.title ("Play Tennis Count")
plt.xlabel("X....Play Decission")
plt.ylabel("Y....Count")
plt.show()
dfx= df[['outlook','temp','humidity','wind']]
dfy= df['play']
dfx
dfy.head()
encoder= LabelEncoder()
for column in dfx.columns:
    dfx[column] = encoder.fit_transform(dfx[column])
dfy=encoder.fit_transform(dfy)
dfy
x_train, x_test, y_train, y_test = train_test_split(
    dfx,
    dfy,
    test_size=0.2,
    random_state=42
)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("Model Saved Successfully")
joblib.dump(model, 'play_model.pkl')
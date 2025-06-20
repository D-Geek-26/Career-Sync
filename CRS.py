import pandas as pd
import joblib as jb
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data=pd.read_excel('Career_Dataset.xlsx')
data.head

# Display all the column names
print("Columns:\n",data.columns)

# Remove unnecessary columns
data=data.drop(['Sr.No.','Student', 's/p'],axis=1,errors='ignore')
print("Columns after dropping:\n",data.columns)

#Function for mapping the ratings based on ranges (POOR,AVG,BEST)
def  map_rating(val):
    if pd.isna(val):
        return 0
    
    try:
        val=float(val)
        if 1<=val<=10:
            return 0
        elif 11<=val<=14:
            return 1
        elif 15<=val<=20:
            return 2
        else:
            return 0

    except (ValueError,TypeError):
        return 0

# Mapping the rating columns
rating_columns=['P1','P2','P3','P4','P5','P6','P7','P8']
for col in rating_columns:
    data[col]=data[col].apply(map_rating)

# Encode the target variable using LabelEncoder
le=LabelEncoder()
data['Job profession']=le.fit_transform(data['Job profession'])

# Define features X and target y
X=data.drop(['Job profession','Course'],axis=1,errors='ignore')
y=data['Job profession']

# Normalize numerical features using StandardScaler
numerical_cols=['Linguistic','Musical','Bodily','Logical - Mathematical','Spatial-Visualization','Interpersonal','Intrapersonal','Naturalist']
scaler=StandardScaler()
X[numerical_cols]=scaler.fit_transform(X[numerical_cols])

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Save the label encoder and scaler for use in prediction
jb.dump(le,'label_encoder.pkl')
jb.dump(scaler,'scaler.pkl')

# Train the Random Forest Classifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# Evaluate the model
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)
print("\nClassification Report:\n",classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))

#Saving the accuracy score to a csv file
accuracy_df=pd.DataFrame({'Metric': ['Accuracy'], 'Value': [accuracy]})
accuracy_df.to_csv('model_accuracy.csv', index=False)


cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", linecolor="Black")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')  #Saving the plot
plt.close()

# Save the trained model
jb.dump(model,'career.pkl')
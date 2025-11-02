# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', None)
# %matplotlib inline

# %%
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(context='notebook', palette='muted', style='whitegrid')



# %%
df = pd.read_csv('data/alzheimers_disease_data.csv')
df.head().T


palette_dict = {
    'Gender': {'Male': "#00bfff", 'Female': '#55efc4'},
    'Smoking': {'Yes': '#00bfff', 'No': '#55efc4'},
    'FamilyHistoryAlzheimers': {'Yes': '#00bfff', 'No': '#55efc4'},
    'CardiovascularDisease': {'Yes': '#00bfff', 'No': '#55efc4'},
    'Diabetes': {'Yes': '#00bfff', 'No': '#55efc4'},
    'Depression': {'Yes': '#00bfff', 'No': '#55efc4'},
    'HeadInjury': {'Yes': '#00bfff', 'No': '#55efc4'},
    'Hypertension': {'Yes': '#00bfff', 'No': '#55efc4'},
    'MemoryComplaints': {'Yes': '#00bfff', 'No': '#55efc4'},
    'BehavioralProblems': {'Yes': '#00bfff', 'No': '#55efc4'},
    'Confusion': {'Yes': '#00bfff', 'No': '#55efc4'},
    'Disorientation': {'Yes': '#00bfff', 'No': '#55efc4'},
    'PersonalityChanges': {'Yes': '#00bfff', 'No': '#55efc4'},
    'DifficultyCompletingTasks': {'Yes': '#00bfff', 'No': '#55efc4'},
    'Forgetfulness': {'Yes': '#00bfff', 'No': '#55efc4'},
    'EducationLevel': {
        'None': "#ffae00",
        'High School': "#fa7000",
        "Bachelor's": "#fff311ff",
        'Higher': "#f80000"
    },
    'Ethnicity': {
        'Caucasian': "#87d7ff",
        'African American': "#0088ff",
        'Asian': "#005de9",
        'Other': "#00c8ff"
    }
}


# %%
df.head()


# %%
df.info()



# %%
df.describe().T


# %%
# Đếm số dòng trùng lặp (duplicate rows)
sum(df.duplicated())

# %%
# Đếm số lần xuất hiện của mỗi giá trị duy nhất trong cột 'DoctorInCharge'
df.DoctorInCharge.value_counts()


# %%
# Xóa các cột không cần thiết khỏi DataFrame
df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)




# Phân tích Dữ Liệu 
# %%
# Xác định các cột số: các cột có hơn 10 giá trị unique được coi là numerical
numerical_columns = [col for col in df.columns if df[col].nunique() > 10]

# Xác định các cột phân loại (categorical): các cột không phải là numerical và không phải là 'Diagnosis'
categorical_columns = df.columns.difference(numerical_columns).difference(['Diagnosis']).to_list()


# Phân phối các tính năng phân loại
# Lập nhãn 
custom_labels = {
    'Gender': ['Male', 'Female'],
    'Ethnicity': ['Caucasian', 'African American', 'Asian', 'Other'],
    'EducationLevel': ['None', 'High School', 'Bachelor\'s', 'Higher'],
    'Smoking': ['No', 'Yes'],
    'FamilyHistoryAlzheimers': ['No', 'Yes'],
    'CardiovascularDisease': ['No', 'Yes'],
    'Diabetes': ['No', 'Yes'],
    'Depression': ['No', 'Yes'],
    'HeadInjury': ['No', 'Yes'],
    'Hypertension': ['No', 'Yes'],
    'MemoryComplaints': ['No', 'Yes'],
    'BehavioralProblems': ['No', 'Yes'],
    'Confusion': ['No', 'Yes'],
    'Disorientation': ['No', 'Yes'],
    'PersonalityChanges': ['No', 'Yes'],
    'DifficultyCompletingTasks': ['No', 'Yes'],
    'Forgetfulness': ['No', 'Yes']
}



# Plot countplots 
for col, labels in custom_labels.items():
    if col in df.columns:
        unique_vals = sorted(df[col].unique())
        if len(unique_vals) == len(labels):
            df[col] = df[col].map({v: lbl for v, lbl in zip(unique_vals, labels)})
        
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    # sns.countplot(data=df, x=column, palette=palette_dict[column])
    # plt.title(f'Countplot of {column}')
    
    # # Directly set custom labels
    # labels = custom_labels[column]
    # ticks = range(len(labels))
    # plt.xticks(ticks=ticks, labels=labels)
    
    # plt.show()
    
    palette = palette_dict.get(column, "muted")

    sns.countplot(data=df, x=column, palette=palette)
    plt.title(f'Countplot of {column}', fontsize=14, fontweight='bold', color='#2c3e50')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    labels = custom_labels.get(column)
    if labels:
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=15)
    else:
        plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.show()



# %%
# Phân phối các tính năng numerical

# Plot histogram 
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.show()
    
    
    
    
    
# Sự tương quan giữa các tính năng

# %%
# Tạo mask
mask = np.triu(np.ones_like(df.corr(), dtype=bool))

# Plot heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(),cmap="coolwarm", cbar_kws={"shrink": .5}, mask=mask)

plt.show()



# %%
# Tính hệ số tương quan Pearson
correlations = df.corr(numeric_only=True)['Diagnosis'][:-1].sort_values()

plt.figure(figsize=(20, 7))

# Tạo biểu đồ thanh của hệ số tương quan Pearson
ax = correlations.plot(kind='bar', width=0.7)

# Đặt giới hạn và nhãn trục y
ax.set(ylim=[-1, 1], ylabel='Pearson Correlation', xlabel='Features', 
        title='Pearson Correlation with Diagnosis')

# Rotate x-axis labels for better readability
ax.set_xticklabels(correlations.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()



# Phân phối target feature

# %%
# Xác định các loại  Response categories và count occurences
categories = [0, 1]
counts = df.Diagnosis.value_counts().tolist()



colors = sns.color_palette("muted")

# Plot the pie chart 
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Diagnosis Distribution')
plt.show()




# Data Preprocessing

# %%
df


# %%
#Giá trị Unique
for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in column '{column}':")
    print(unique_values)
    print()
    
    
# %%
columns = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']

#normalize các cột
min_max_scaler = MinMaxScaler()
df[columns] = min_max_scaler.fit_transform(df[columns])

#standardize các cột
standard_scaler = StandardScaler()
df[columns] = standard_scaler.fit_transform(df[columns])


# %%
ethnicity_encoded = pd.get_dummies(df['Ethnicity'], prefix='Ethnicity')

# Nối đặc trưng mới vào df và bỏ cột cũ 'Ethnicity'
df = pd.concat([df.drop(columns=['Ethnicity']), ethnicity_encoded], axis=1)


# %%
df


# Modeling

# %%
#chia dữ liệu thành các feature và target
X = df.drop(columns = ['Diagnosis'])
y = df['Diagnosis']

#chia dữ liệu thành các tập huấn luyện và thử nghiệm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

#xác định hyperparameter grids 
param_grids = {
    'Decision Tree': {'max_depth': [3, 5, 7, 12, None]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale', 'auto']},
}

#khởi tạo model
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
}

#fit models 
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv = 5, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(f'{name} Classification Report:\n{report}\nBest Parameters: {grid_search.best_params_}\n')

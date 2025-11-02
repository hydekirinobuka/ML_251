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

sns.set_theme(context='notebook', palette='muted', style='darkgrid')


# %%
# Đọc dữ liệu từ file CSV
df = pd.read_csv('data/alzheimers_disease_data.csv')
df.head().T


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
# Đếm số lần xuất hiện của mỗi giá trị trong cột 'DoctorInCharge'
df.DoctorInCharge.value_counts()

# %%
# Xóa các cột không cần thiết
df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)


# -------------------------------------------------------------
# PHÂN TÍCH DỮ LIỆU
# -------------------------------------------------------------

# %%
# Xác định các cột số (numerical): có hơn 10 giá trị khác nhau
numerical_columns = [col for col in df.columns if df[col].nunique() > 10]

# Xác định các cột phân loại (categorical): không nằm trong numerical và không phải cột 'Diagnosis'
categorical_columns = df.columns.difference(numerical_columns).difference(['Diagnosis']).to_list()


# Tạo nhãn hiển thị cho các giá trị phân loại
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

# %%
# Vẽ biểu đồ countplot cho các cột phân loại
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column)
    plt.title(f'Countplot of {column}')
    
    # Thiết lập nhãn hiển thị (ví dụ: Yes / No)
    labels = custom_labels[column]
    ticks = range(len(labels))
    plt.xticks(ticks=ticks, labels=labels)
    
    plt.show()


# -------------------------------------------------------------
# PHÂN PHỐI CÁC THUỘC TÍNH DẠNG SỐ
# -------------------------------------------------------------

# %%
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.show()
    
    
# -------------------------------------------------------------
# MA TRẬN TƯƠNG QUAN GIỮA CÁC ĐẶC TRƯNG
# -------------------------------------------------------------

# %%
mask = np.triu(np.ones_like(df.corr(), dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", cbar_kws={"shrink": .5}, mask=mask)
plt.show()


# %%
# Tính hệ số tương quan Pearson với biến mục tiêu (Diagnosis)
correlations = df.corr(numeric_only=True)['Diagnosis'][:-1].sort_values()

plt.figure(figsize=(20, 7))
ax = correlations.plot(kind='bar', width=0.7)
ax.set(ylim=[-1, 1], ylabel='Hệ số tương quan Pearson', xlabel='Đặc trưng', 
        title='Tương quan giữa các đặc trưng và Diagnosis')
ax.set_xticklabels(correlations.index, rotation=45, ha='right')
plt.tight_layout()
plt.show()


# -------------------------------------------------------------
# PHÂN PHỐI NHÃN ĐÍCH (Diagnosis)
# -------------------------------------------------------------

# %%
categories = [0, 1]
counts = df.Diagnosis.value_counts().tolist()

colors = sns.color_palette("muted")

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Phân bố Diagnosis')
plt.show()


# -------------------------------------------------------------
# TIỀN XỬ LÝ DỮ LIỆU
# -------------------------------------------------------------

# %%
# Hiển thị toàn bộ dữ liệu
df

# %%
# Hiển thị các giá trị duy nhất trong từng cột
for column in df.columns:
    unique_values = df[column].unique()
    print(f"Các giá trị duy nhất trong cột '{column}':")
    print(unique_values)
    print()
    
# %%
# Danh sách các cột dạng số cần chuẩn hóa
columns = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
           'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
           'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']

# Chuẩn hóa (0–1)
min_max_scaler = MinMaxScaler()
df[columns] = min_max_scaler.fit_transform(df[columns])

# Chuẩn hóa về phân phối chuẩn (Z-score)
standard_scaler = StandardScaler()
df[columns] = standard_scaler.fit_transform(df[columns])


# %%
# Mã hóa one-hot cho cột 'Ethnicity'
ethnicity_encoded = pd.get_dummies(df['Ethnicity'], prefix='Ethnicity')

# Gộp lại vào DataFrame và loại bỏ cột gốc
df = pd.concat([df.drop(columns=['Ethnicity']), ethnicity_encoded], axis=1)

# %%
df


# -------------------------------------------------------------
# MÔ HÌNH HÓA (MODELING)
# -------------------------------------------------------------

# %%
# Tách dữ liệu thành đặc trưng (X) và nhãn (y)
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Định nghĩa lưới tham số (hyperparameter grid) cho từng mô hình
param_grids = {
    'Decision Tree': {'max_depth': [3, 5, 7, 12, None]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale', 'auto']},
}

# Khởi tạo các mô hình phân loại
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
}

# Huấn luyện mô hình và tinh chỉnh siêu tham số bằng GridSearchCV
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(f'{name} – Báo cáo phân loại:\n{report}\nThông số tốt nhất: {grid_search.best_params_}\n')

# %%

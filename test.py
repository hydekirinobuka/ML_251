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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, C=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true, y_pred, weights):
        m = len(y_true)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        if self.regularization == 'l2':
            lambda_param = 1 / (2 * self.C * m)
            l2_penalty = lambda_param * np.sum(weights ** 2)
            loss += l2_penalty
        
        return loss
    
    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            if self.regularization == 'l2':
                lambda_param = 1 / (self.C * n_samples)
                dw += lambda_param * self.weights
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                loss = self._compute_loss(y, y_predicted, self.weights)
                self.losses.append(loss)
        
        return self
    
    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'regularization': self.regularization,
            'C': self.C
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

pd.set_option('display.max_columns', None)
# %matplotlib inline

# %%
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(context='notebook', palette='muted', style='darkgrid')



# Đọc dữ liệu từ file CSV
df = pd.read_csv('data/alzheimers_disease_data.csv')
df.head().T



df.head()


df.info()


df.describe().T


# Đếm số dòng trùng lặp (duplicate rows)
sum(df.duplicated())


# Đếm số lần xuất hiện của mỗi giá trị trong cột 'DoctorInCharge'
df.DoctorInCharge.value_counts()


# Xóa các cột không cần thiết
df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)


# -------------------------------------------------------------
# PHÂN TÍCH DỮ LIỆU
# -------------------------------------------------------------


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


for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.show()
    
    
# -------------------------------------------------------------
# MA TRẬN TƯƠNG QUAN GIỮA CÁC ĐẶC TRƯNG
# -------------------------------------------------------------


mask = np.triu(np.ones_like(df.corr(), dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", cbar_kws={"shrink": .5}, mask=mask)
plt.show()



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


# Hiển thị toàn bộ dữ liệu
df


# Hiển thị các giá trị duy nhất trong từng cột
for column in df.columns:
    unique_values = df[column].unique()
    print(f"Các giá trị duy nhất trong cột '{column}':")
    print(unique_values)
    print()
    

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



# Mã hóa one-hot cho cột 'Ethnicity'
ethnicity_encoded = pd.get_dummies(df['Ethnicity'], prefix='Ethnicity')

# Gộp lại vào DataFrame và loại bỏ cột gốc
df = pd.concat([df.drop(columns=['Ethnicity']), ethnicity_encoded], axis=1)


df


# -------------------------------------------------------------
# MÔ HÌNH HÓA (MODELING)
# -------------------------------------------------------------


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


# %%
# -------------------------------------------------------------
# TEST CHI TIẾT LOGISTIC REGRESSION VỚI CÁC THAM SỐ KHÁC NHAU 
# n_iterations = 1000
# -------------------------------------------------------------


print("\n" + "=" * 70)
print("TEST CHI TIẾT LOGISTIC REGRESSION VỚI CÁC CẤU HÌNH KHÁC NHAU n_iterations = 1000")
print("=" * 70)

test_configs = [
    {'learning_rate': 0.01, 'n_iterations': 1000, 'regularization': None, 'C': 1.0},
    {'learning_rate': 0.01, 'n_iterations': 1000, 'regularization': 'l2', 'C': 0.1},
    {'learning_rate': 0.01, 'n_iterations': 1000, 'regularization': 'l2', 'C': 1.0},
    {'learning_rate': 0.01, 'n_iterations': 1000, 'regularization': 'l2', 'C': 10.0},
]

results = []

for idx, config in enumerate(test_configs, 1):
    print(f"\n{'-' * 70}")
    print(f"Cấu hình {idx}:")
    print(f"  learning_rate={config['learning_rate']}, n_iterations={config['n_iterations']}")
    print(f"  regularization={config['regularization']}, C={config['C']}")
    print(f"{'-' * 70}")
    
    # Huấn luyện mô hình
    model = LogisticRegression(**config)
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(cm)
    print(f"TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")
    
    # Hiển thị loss qua các iterations
    print(f"\nLoss theo các iterations (5 giá trị cuối):")
    for i, loss in enumerate(model.losses[-5:]):
        iteration_num = (len(model.losses) - 5 + i) * 100
        print(f"  Iteration {iteration_num}: Loss = {loss:.6f}")
    
    results.append({
        'config': config,
        'accuracy': accuracy,
        'recall': recall,
        'specificity': specificity,
        'model': model
    })

# Tìm cấu hình tốt nhất
best_result = max(results, key=lambda x: x['accuracy'])
print("\n" + "=" * 70)
print("KẾT QUẢ TỐT NHẤT")
print("=" * 70)
print(f"Cấu hình: {best_result['config']}")
print(f"Accuracy: {best_result['accuracy']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")
print(f"Specificity: {best_result['specificity']:.4f}")

print("\n✓ Hoàn tất phân tích và đánh giá mô hình!")




# %%
# -------------------------------------------------------------
# TEST CHI TIẾT LOGISTIC REGRESSION VỚI CÁC THAM SỐ KHÁC NHAU 
# n_iterations = 10000
# -------------------------------------------------------------


print("\n" + "=" * 70)
print("TEST CHI TIẾT LOGISTIC REGRESSION VỚI CÁC CẤU HÌNH KHÁC NHAU n_iterations = 10000")
print("=" * 70)

test_configs = [
    {'learning_rate': 0.01, 'n_iterations': 10000, 'regularization': None, 'C': 1.0},
    {'learning_rate': 0.01, 'n_iterations': 10000, 'regularization': 'l2', 'C': 0.1},
    {'learning_rate': 0.01, 'n_iterations': 10000, 'regularization': 'l2', 'C': 1.0},
    {'learning_rate': 0.01, 'n_iterations': 10000, 'regularization': 'l2', 'C': 10.0},
]

results = []

for idx, config in enumerate(test_configs, 1):
    print(f"\n{'-' * 70}")
    print(f"Cấu hình {idx}:")
    print(f"  learning_rate={config['learning_rate']}, n_iterations={config['n_iterations']}")
    print(f"  regularization={config['regularization']}, C={config['C']}")
    print(f"{'-' * 70}")
    
    # Huấn luyện mô hình
    model = LogisticRegression(**config)
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(cm)
    print(f"TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")
    
    # Hiển thị loss qua các iterations
    print(f"\nLoss theo các iterations (5 giá trị cuối):")
    for i, loss in enumerate(model.losses[-5:]):
        iteration_num = (len(model.losses) - 5 + i) * 100
        print(f"  Iteration {iteration_num}: Loss = {loss:.6f}")
    
    results.append({
        'config': config,
        'accuracy': accuracy,
        'recall': recall,
        'specificity': specificity,
        'model': model
    })

# Tìm cấu hình tốt nhất
best_result = max(results, key=lambda x: x['accuracy'])
print("\n" + "=" * 70)
print("KẾT QUẢ TỐT NHẤT")
print("=" * 70)
print(f"Cấu hình: {best_result['config']}")
print(f"Accuracy: {best_result['accuracy']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")
print(f"Specificity: {best_result['specificity']:.4f}")

print("\n✓ Hoàn tất phân tích và đánh giá mô hình!")



#%%
# Huấn luyện mô hình và tinh chỉnh siêu tham số bằng GridSearchCV


# Decision Tree
dt_grid = GridSearchCV(DecisionTreeClassifier(), param_grids['Decision Tree'], cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_
dt_pred = dt_best.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
dt_recall = dt_cm[1,1] / (dt_cm[1,1] + dt_cm[1,0])
dt_specificity = dt_cm[0,0] / (dt_cm[0,0] + dt_cm[0,1])
print(f"Decision Tree: TP={dt_cm[1,1]}, TN={dt_cm[0,0]}, FP={dt_cm[0,1]}, FN={dt_cm[1,0]}")
disp = ConfusionMatrixDisplay.from_estimator(dt_best, X_test, y_test)
plt.title("Confusion Matrix - Decision Tree")
plt.savefig("latex/assets/confusion_matrix_decisiontree.png", dpi=300, bbox_inches='tight')
plt.close()


# KNN: accuracy theo từng K (1-15)
knn_acc_list = []
knn_k_list = list(range(1, 16))
for k in knn_k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    knn_acc_list.append(acc)
# K tối ưu
knn_best_k = knn_k_list[np.argmax(knn_acc_list)]
knn_best = KNeighborsClassifier(n_neighbors=knn_best_k)
knn_best.fit(X_train, y_train)
knn_pred = knn_best.predict(X_test)
knn_cm = confusion_matrix(y_test, knn_pred)
knn_acc = accuracy_score(y_test, knn_pred)
knn_recall = knn_cm[1,1] / (knn_cm[1,1] + knn_cm[1,0])
knn_specificity = knn_cm[0,0] / (knn_cm[0,0] + knn_cm[0,1])
print(f"KNN (K={knn_best_k}): TP={knn_cm[1,1]}, TN={knn_cm[0,0]}, FP={knn_cm[0,1]}, FN={knn_cm[1,0]}")
plt.plot(knn_k_list, knn_acc_list, marker='o')
plt.xlabel('Số láng giềng K')
plt.ylabel('Accuracy')
plt.title('Độ chính xác KNN theo số lượng K')
plt.savefig('latex/assets/knn_k_vs_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()


# Logistic Regression
logreg_grid = GridSearchCV(LogisticRegression(), param_grids['Logistic Regression'], cv=5, scoring='accuracy')
logreg_grid.fit(X_train, y_train)
logreg_best = logreg_grid.best_estimator_
logreg_pred = logreg_best.predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_pred)
logreg_cm = confusion_matrix(y_test, logreg_pred)
logreg_recall = logreg_cm[1,1] / (logreg_cm[1,1] + logreg_cm[1,0])
logreg_specificity = logreg_cm[0,0] / (logreg_cm[0,0] + logreg_cm[0,1])
print(f"Logistic Regression: TP={logreg_cm[1,1]}, TN={logreg_cm[0,0]}, FP={logreg_cm[0,1]}, FN={logreg_cm[1,0]}")
coefs = logreg_best.weights
features = X.columns
plt.figure(figsize=(8,6))
plt.barh(features, coefs)
plt.xlabel('Hệ số hồi quy')
plt.title('Hệ số hồi quy các biến độc lập')
plt.tight_layout()
plt.savefig('latex/assets/logreg_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()


# SVM: so sánh accuracy các kernel
svm_acc_list = []
svm_kernel_names = ['linear', 'rbf']
svm_results = {}
for kernel in svm_kernel_names:
    svm = SVC(kernel=kernel, C=1, gamma='scale')
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    print(f"SVM ({kernel}): TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")
    svm_acc_list.append(acc)
    svm_results[kernel] = {'acc': acc, 'recall': recall, 'specificity': specificity}
plt.bar(svm_kernel_names, svm_acc_list, color=['skyblue', 'orange'])
plt.ylabel('Accuracy')
plt.title('So sánh accuracy SVM với các kernel')
plt.savefig('latex/assets/svm_kernel_compare.png', dpi=300, bbox_inches='tight')
plt.close()


# Lưu số liệu ra file txt để đối chiếu
with open('result.txt', 'w', encoding='utf-8') as f:
    f.write(f"Decision Tree: accuracy={dt_acc:.3f}, recall={dt_recall:.3f}, specificity={dt_specificity:.3f}\n")
    f.write(f"KNN (best k={knn_best_k}): accuracy={knn_acc:.3f}, recall={knn_recall:.3f}, specificity={knn_specificity:.3f}\n")
    f.write(f"KNN acc list: {[round(a,3) for a in knn_acc_list]}\n")
    f.write(f"Logistic Regression: accuracy={logreg_acc:.3f}, recall={logreg_recall:.3f}, specificity={logreg_specificity:.3f}\n")
    for kernel in svm_kernel_names:
        r = svm_results[kernel]
        f.write(f"SVM ({kernel}): accuracy={r['acc']:.3f}, recall={r['recall']:.3f}, specificity={r['specificity']:.3f}\n")


# %%
#fit models using GridSearchCV for hyperparameter tuning
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv = 5, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(f'{name} Classification Report:\n{report}\nBest Parameters: {grid_search.best_params_}\n')




# %%
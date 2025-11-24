# Test mô hình Logistic Regression tự xây dựng
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION
# -------------------------------------------------------------

class LogisticRegression:
    """
    Mô hình Logistic Regression được xây dựng từ đầu
    Sử dụng Gradient Descent để tối ưu hóa
    """
    
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


# -------------------------------------------------------------
# LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU
# -------------------------------------------------------------

print("=" * 70)
print("LOAD DỮ LIỆU ALZHEIMER'S DISEASE")
print("=" * 70)

df = pd.read_csv('data/alzheimers_disease_data.csv')
print(f"\nKích thước dữ liệu: {df.shape}")
print(f"Số dòng trùng lặp: {sum(df.duplicated())}")

# Xóa các cột không cần thiết
df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

# Chuẩn hóa các cột số
columns = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
           'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
           'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']

min_max_scaler = MinMaxScaler()
df[columns] = min_max_scaler.fit_transform(df[columns])

standard_scaler = StandardScaler()
df[columns] = standard_scaler.fit_transform(df[columns])

# Mã hóa one-hot cho cột 'Ethnicity'
ethnicity_encoded = pd.get_dummies(df['Ethnicity'], prefix='Ethnicity')
df = pd.concat([df.drop(columns=['Ethnicity']), ethnicity_encoded], axis=1)

print(f"Kích thước sau tiền xử lý: {df.shape}")

# -------------------------------------------------------------
# CHIA DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH
# -------------------------------------------------------------

print("\n" + "=" * 70)
print("HUẤN LUYỆN MÔ HÌNH LOGISTIC REGRESSION TỰ XÂY DỰNG")
print("=" * 70)

X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nKích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# Test với các tham số khác nhau
test_configs = [
    {'learning_rate': 0.01, 'n_iterations': 1000, 'regularization': None, 'C': 1.0},
    {'learning_rate': 0.01, 'n_iterations': 1000, 'regularization': 'l2', 'C': 0.1},
    {'learning_rate': 0.01, 'n_iterations': 1000, 'regularization': 'l2', 'C': 1.0},
    {'learning_rate': 0.01, 'n_iterations': 1000, 'regularization': 'l2', 'C': 10.0},
]

results = []

for config in test_configs:
    print(f"\n{'-' * 70}")
    print(f"Cấu hình: learning_rate={config['learning_rate']}, n_iterations={config['n_iterations']}")
    print(f"          regularization={config['regularization']}, C={config['C']}")
    print(f"{'-' * 70}")
    
    # Huấn luyện mô hình
    model = LogisticRegression(**config)
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Hiển thị loss qua các iterations
    print(f"\nLoss theo các iterations (mỗi 100 iterations):")
    for i, loss in enumerate(model.losses[-5:]):  # Hiển thị 5 giá trị cuối
        print(f"  Iteration {(len(model.losses) - 5 + i) * 100}: Loss = {loss:.6f}")
    
    results.append({
        'config': config,
        'accuracy': accuracy,
        'model': model
    })

# Tìm cấu hình tốt nhất
best_result = max(results, key=lambda x: x['accuracy'])
print("\n" + "=" * 70)
print("KẾT QUẢ TỐT NHẤT")
print("=" * 70)
print(f"Cấu hình: {best_result['config']}")
print(f"Accuracy: {best_result['accuracy']:.4f}")

print("\n✓ Hoàn tất kiểm tra mô hình Logistic Regression tự xây dựng!")

# --- CÀI ĐẶT THƯ VIỆN CẦN THIẾT ---
import pandas as pd
import numpy as np
import time
import lightgbm as lgb 
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

# Các mô hình khác
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

print("--- Bắt đầu chương trình: So sánh LightGBM, XGBoost và Ridge ---")

# ======================================================================
# BƯỚC 1 & 2: TẢI VÀ LỌC DỮ LIỆU
# ======================================================================

file_name = 'D:\\XGB\\postings.csv'
try:
    df = pd.read_csv(file_name)
    print(f"Tải dữ liệu thành công. Shape: {df.shape}")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy tệp '{file_name}'.")
    raise

target_column = 'normalized_salary'
df_filtered = df.dropna(subset=[target_column])
df_filtered = df_filtered[
    (df_filtered[target_column] >= 10000.0) &
    (df_filtered[target_column] <= 1000000.0)
].copy()

feature_columns = [
    'company_name', 'title', 'description', 'location', 'views',
    'formatted_work_type', 'remote_allowed', 'formatted_experience_level',
    'skills_desc', 'sponsored', 'application_type'
]
existent_feature_columns = [col for col in feature_columns if col in df_filtered.columns]
df_clean = df_filtered[existent_feature_columns + [target_column]].copy()

# ======================================================================
# BƯỚC 3 & 4: PIPELINE TIỀN XỬ LÝ
# ======================================================================

if 'description' in df_clean.columns:
    df_clean['description'] = df_clean['description'].fillna("")
if 'skills_desc' in df_clean.columns:
    df_clean['skills_desc'] = df_clean['skills_desc'].fillna("")
df_clean['all_text'] = df_clean['description'] + " " + df_clean['skills_desc']

def convert_to_string(df):
    return df.astype(str)

text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=500, stop_words='english', dtype=np.float32))
])
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
high_card_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('to_str', FunctionTransformer(convert_to_string)),
    ('encoder', OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist'))
])
low_card_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('to_str', FunctionTransformer(convert_to_string)),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'all_text'),
        ('num', numeric_transformer, [c for c in ['views'] if c in df_clean.columns]),
        ('high_card', high_card_transformer, [c for c in ['title', 'location', 'company_name'] if c in df_clean.columns]),
        ('low_card', low_card_transformer, [c for c in ['formatted_work_type', 'remote_allowed', 'formatted_experience_level'] if c in df_clean.columns])
    ],
    remainder='drop'
)
# ======================================================================
# BƯỚC 5: HUẤN LUYỆN VÀ SO SÁNH 
# ======================================================================
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

y = df_clean[target_column]
X = df_clean.drop(columns=[target_column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Định nghĩa mô hình ---
lgbm_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1 
)

models = {
    "Ridge (Baseline)": Pipeline([('pre', preprocessor), ('m', Ridge(random_state=42))]),
    "LightGBM": Pipeline([('pre', preprocessor), ('m', lgbm_model)]),
    "XGBoost": Pipeline([('pre', preprocessor), ('m', XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42))])
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Bắt đầu so sánh hiệu suất (K-Fold = 5) ---")
results = {}

for name, pipeline in models.items():
    print(f"-> Đang chạy: {name}...")
    start_time = time.time()

    # 1. K-Fold Cross Validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2', n_jobs=1)

    # 2. Test độc lập
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    end_time = time.time()

    results[name] = {
        "Time (s)": end_time - start_time,
        "CV Mean R2": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Test MAE": mean_absolute_error(y_test, y_pred)
    }

# --- HIỂN THỊ KẾT QUẢ ---
print("\n--- BẢNG XẾP HẠNG MÔ HÌNH ---")
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by=["CV Mean R2", "Time (s)"], ascending=[False, True])

try:
    print(results_df.to_markdown(floatfmt=".4f"))
except ImportError:
    print(results_df)

from sklearn.model_selection import RandomizedSearchCV

print("--- BẮT ĐẦU TỐI ƯU HÓA THAM SỐ (TUNING) CHO LIGHTGBM ---")

# 1. Định nghĩa không gian tham số 
param_dist = {
    'model__n_estimators': [200, 500, 800],      # Số lượng cây
    'model__learning_rate': [0.01, 0.05, 0.1],   # Tốc độ học
    'model__num_leaves': [31, 50, 70],           # Độ phức tạp của lá
    'model__max_depth': [5, 7, 10, -1],          # Độ sâu tối đa (-1 là không giới hạn)
    'model__min_child_samples': [20, 50, 100]    # Số lượng mẫu tối thiểu trên 1 lá (chống nhiễu)
}

# 2. Tạo Pipeline cho LightGBM
pipeline_lgbm = Pipeline([
    ('pre', preprocessor),
    ('model', LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))
])

# 3. Thiết lập tìm kiếm ngẫu nhiên
# n_iter=20: Máy sẽ chọn ngẫu nhiên 20 tổ hợp để thử
search = RandomizedSearchCV(
    pipeline_lgbm,
    param_distributions=param_dist,
    n_iter=20,
    cv=3, # Chia 3 phần để kiểm tra chéo
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# 4. Bắt đầu huấn luyện 
print("Đang tìm kiếm bộ tham số tốt nhất...")
start_time = time.time()

# Lọc cảnh báo để màn hình sạch sẽ
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    search.fit(X_train, y_train)

end_time = time.time()
print(f"Hoàn tất sau: {end_time - start_time:.2f} giây")

# 5. Kết quả tốt nhất
print("\n--- KẾT QUẢ TỐI ƯU ---")
print(f"Điểm R2 tốt nhất trên tập Train (CV): {search.best_score_:.4f}")
print("Bộ tham số tốt nhất:")
for param, value in search.best_params_.items():
    print(f"  - {param}: {value}")

# 6. Kiểm tra lại trên tập Test 
best_model = search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
r2_tuned = r2_score(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print(f"\nHiệu suất trên tập TEST (Thực tế):")
print(f"  - R2 Score: {r2_tuned:.4f}")
print(f"  - MAE: {mae_tuned:,.0f}")

# ======================================================================
# BƯỚC 6: LƯU MÔ HÌNH VÀ KẾT QUẢ
# ======================================================================

import joblib
import json
from datetime import datetime

def save_model(model, preprocessor, results, output_dir='models'):
    """
    Lưu mô hình đã huấn luyện, preprocessor và kết quả vào thư mục
    
    Parameters:
        model : Pipeline hoặc model object
        preprocessor : ColumnTransformer
        results : dict  -> chứa r2_score, mae, cv_mean_r2, best_params
        output_dir : str
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Lưu mô hình LightGBM (đã nằm trong Pipeline)
    model_path = os.path.join(output_dir, f"lightgbm_model_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"✓ Mô hình lưu tại: {model_path}")

    # 2. Lưu preprocessor (có thể tách để tái sử dụng)
    pre_path = os.path.join(output_dir, f"preprocessor_{timestamp}.pkl")
    joblib.dump(preprocessor, pre_path)
    print(f"✓ Preprocessor lưu tại: {pre_path}")

    # 3. Lưu kết quả tuning + test vào JSON
    results_path = os.path.join(output_dir, f"results_{timestamp}.json")
    results['timestamp'] = timestamp  # thêm thời gian tạo record

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✓ Kết quả lưu tại: {results_path}")

    # 4. Lưu bản TXT mô tả model
    info_path = os.path.join(output_dir, f"model_info_{timestamp}.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=== THÔNG TIN MÔ HÌNH LIGHTGBM ===\n\n")
        f.write(f"Thời gian tạo: {timestamp}\n\n")

        f.write("=== KẾT QUẢ TEST ===\n")
        f.write(f"R2 Score: {results['r2_score']:.4f}\n")
        f.write(f"MAE: {results['mae']:,.0f}\n\n")

        f.write(f"CV Mean R2: {results['cv_mean_r2']:.4f}\n\n")

        f.write("=== BỘ THAM SỐ TỐI ƯU ===\n")
        for param, value in results["best_params"].items():
            f.write(f"{param}: {value}\n")

    print(f"✓ Thông tin mô hình lưu tại: {info_path}")

    return {
        'model_path': model_path,
        'preprocessor_path': pre_path,
        'results_path': results_path,
        'info_path': info_path
    }
# Gọi hàm lưu mô hình
# --- Tạo biến results_to_save để truyền vào hàm save_model ---
results_to_save = {
    'r2_score': float(r2_tuned),
    'mae': float(mae_tuned),
    'cv_mean_r2': float(search.best_score_),
    'best_params': search.best_params_
}

print("\n--- DANG LUU MO HINH ---")
saved_paths = save_model(best_model, preprocessor, results_to_save)
print("\nHoan tat!")

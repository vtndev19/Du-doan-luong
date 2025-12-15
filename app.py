# --- CÀI ĐẶT THƯ VIỆN CẦN THIẾT ---
import pandas as pd
import numpy as np
import time
import lightgbm as lgb 
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform  # Thư viện sinh số ngẫu nhiên cho Tuning

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

# Các mô hình khác
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

import joblib
import json
from datetime import datetime
import os
import warnings

# Tắt cảnh báo để màn hình sạch sẽ hơn
warnings.filterwarnings("ignore")

print("--- Bắt đầu chương trình: So sánh LightGBM, XGBoost và Ridge ---")

# ======================================================================
# BƯỚC 1 & 2: TẢI VÀ LỌC DỮ LIỆU
# ======================================================================

file_name = 'D:\\XGB\\postings.csv' # Đường dẫn file của bạn
try:
    df = pd.read_csv(file_name)
    print(f"Tải dữ liệu thành công. Shape: {df.shape}")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy tệp '{file_name}'.")
    # Tạo dữ liệu giả lập để test code nếu không có file thật (Comment dòng dưới nếu chạy thật)
    # raise
    df = pd.DataFrame() 

target_column = 'normalized_salary'

# Kiểm tra nếu dữ liệu tồn tại
if not df.empty and target_column in df.columns:
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
else:
    print("Cảnh báo: Đang chạy demo hoặc không tìm thấy cột target.")
    # Dữ liệu demo dự phòng nếu không đọc được file
    df_clean = pd.DataFrame({
        'title': ['Data Scientist', 'Engineer', 'Analyst'] * 100,
        'description': ['python sql', 'java c++', 'excel powerbi'] * 100,
        'location': ['NY', 'CA', 'TX'] * 100,
        'normalized_salary': np.random.randint(50000, 150000, 300)
    })
    df_clean['skills_desc'] = df_clean['description']
    df_clean['all_text'] = df_clean['description']

# ======================================================================
# BƯỚC 3 & 4: PIPELINE TIỀN XỬ LÝ
# ======================================================================

if 'description' in df_clean.columns:
    df_clean['description'] = df_clean['description'].fillna("")
if 'skills_desc' in df_clean.columns:
    df_clean['skills_desc'] = df_clean['skills_desc'].fillna("")

# Tạo cột text gộp
if 'all_text' not in df_clean.columns:
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

# Xác định cột nào là số, cột nào là danh mục dựa trên dữ liệu thực tế
num_cols = [c for c in ['views'] if c in df_clean.columns]
high_card_cols = [c for c in ['title', 'location', 'company_name'] if c in df_clean.columns]
low_card_cols = [c for c in ['formatted_work_type', 'remote_allowed', 'formatted_experience_level'] if c in df_clean.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'all_text'),
        ('num', numeric_transformer, num_cols),
        ('high_card', high_card_transformer, high_card_cols),
        ('low_card', low_card_transformer, low_card_cols)
    ],
    remainder='drop'
)

# ======================================================================
# BƯỚC 5: HUẤN LUYỆN VÀ SO SÁNH SƠ BỘ
# ======================================================================

y = df_clean[target_column]
X = df_clean.drop(columns=[target_column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Định nghĩa mô hình cơ bản ---
lgbm_model = LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)

models = {
    "Ridge (Baseline)": Pipeline([('pre', preprocessor), ('m', Ridge(random_state=42))]),
    "LightGBM (Base)": Pipeline([('pre', preprocessor), ('m', lgbm_model)]),
    "XGBoost": Pipeline([('pre', preprocessor), ('m', XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42))])
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Bắt đầu so sánh hiệu suất sơ bộ (K-Fold = 5) ---")
results = {}

for name, pipeline in models.items():
    print(f"-> Đang chạy: {name}...")
    start_time = time.time()
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2', n_jobs=1)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    end_time = time.time()
    
    results[name] = {
        "Time (s)": end_time - start_time,
        "CV Mean R2": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Test MAE": mean_absolute_error(y_test, y_pred)
    }

print("\n--- BẢNG XẾP HẠNG MÔ HÌNH SƠ BỘ ---")
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by=["CV Mean R2", "Time (s)"], ascending=[False, True])
try:
    print(results_df.to_markdown(floatfmt=".4f"))
except:
    print(results_df)

# ======================================================================
# BƯỚC 5 (NÂNG CAO): TỐI ƯU HÓA THAM SỐ (ADVANCED TUNING)
# ======================================================================

print("\n--- BẮT ĐẦU TỐI ƯU HÓA CHUYÊN SÂU (ADVANCED TUNING) CHO LIGHTGBM ---")

# 1. Định nghĩa không gian tham số mở rộng
# Sử dụng phân phối xác suất (randint, uniform) để quét không gian tốt hơn
param_dist = {
    # -- Cấu trúc cây (Tree Structure) --
    'model__n_estimators': [200, 500, 800, 1000, 1500],  # Tăng số lượng cây
    'model__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1], # Thử nghiệm learning rate nhỏ hơn
    'model__num_leaves': randint(20, 150),         # Random độ phức tạp lá từ 20 đến 150
    'model__max_depth': [-1, 7, 10, 15, 20],       # Độ sâu (-1 là không giới hạn)
    'model__min_child_samples': randint(10, 100),  # Số mẫu tối thiểu trên lá
    
    # -- Chống Overfitting (Regularization) --
    'model__reg_alpha': [0, 0.01, 0.1, 1, 5, 10],   # L1 Regularization
    'model__reg_lambda': [0, 0.01, 0.1, 1, 5, 10],  # L2 Regularization
    
    # -- Lấy mẫu ngẫu nhiên (Subsampling) --
    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],      # Chỉ dùng % dữ liệu mỗi lần build cây
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0] # Chỉ dùng % features mỗi lần build cây
}

# 2. Tạo Pipeline cho LightGBM cần tối ưu
pipeline_lgbm_tuned = Pipeline([ghit
    ('pre', preprocessor),
    ('model', LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))
])

# 3. Thiết lập tìm kiếm ngẫu nhiên (RandomizedSearchCV)
# TĂNG SỐ LƯỢNG ITERATIONS LÊN 50
ITERATIONS = 50

search = RandomizedSearchCV(
    pipeline_lgbm_tuned,
    param_distributions=param_dist,
    n_iter=ITERATIONS,   # Thử nghiệm 50 tổ hợp ngẫu nhiên
    cv=3,                # Giữ 3-Fold để cân bằng tốc độ
    scoring='r2',
    verbose=1,           # Hiển thị tiến trình
    random_state=42,
    n_jobs=-1
)

# 4. Bắt đầu quá trình Tuning
print(f"Đang tìm kiếm bộ tham số tốt nhất với {ITERATIONS} tổ hợp...")
start_time = time.time()

search.fit(X_train, y_train)

end_time = time.time()
duration = end_time - start_time
print(f"Hoàn tất quá trình Tuning sau: {duration:.2f} giây ({duration/60:.2f} phút)")

# 5. Kết quả tối ưu
print("\n--- KẾT QUẢ TỐI ƯU ---")
print(f"Điểm R2 tốt nhất trên tập Train (CV): {search.best_score_:.4f}")
print("Bộ tham số tốt nhất:")
for param, value in search.best_params_.items():
    print(f"  - {param}: {value}")

# 6. Kiểm tra lại trên tập Test (Dữ liệu chưa từng thấy)
best_model = search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
r2_tuned = r2_score(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print(f"\nHiệu suất trên tập TEST (Thực tế):")
print(f"  - R2 Score: {r2_tuned:.4f}")
print(f"  - MAE: {mae_tuned:,.0f}")

# So sánh với mô hình cơ bản ban đầu
base_r2 = results["LightGBM (Base)"]["Test MAE"]
improvement = base_r2 - mae_tuned
print(f"  -> Cải thiện MAE so với bản chưa tune: {improvement:,.0f} (Thấp hơn là tốt hơn)")

# ======================================================================
# BƯỚC 6: LƯU MÔ HÌNH VÀ KẾT QUẢ
# ======================================================================

def save_model(model, preprocessor, results, output_dir='models'):
    """
    Lưu mô hình đã huấn luyện, preprocessor và kết quả vào thư mục
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Lưu mô hình LightGBM (Lưu cả pipeline chứa preprocessor và model)
    model_path = os.path.join(output_dir, f"lightgbm_tuned_model_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"✓ Mô hình lưu tại: {model_path}")

    # 2. Lưu preprocessor riêng (để tái sử dụng nếu cần)
    pre_path = os.path.join(output_dir, f"preprocessor_{timestamp}.pkl")
    joblib.dump(preprocessor, pre_path)
    
    # 3. Lưu kết quả tuning + test vào JSON
    results_path = os.path.join(output_dir, f"results_{timestamp}.json")
    results['timestamp'] = timestamp

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # 4. Lưu bản TXT mô tả model
    info_path = os.path.join(output_dir, f"model_info_{timestamp}.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=== THÔNG TIN MÔ HÌNH LIGHTGBM (TUNED) ===\n\n")
        f.write(f"Thời gian tạo: {timestamp}\n")
        f.write(f"Số lần thử nghiệm (n_iter): {ITERATIONS}\n\n")
        
        f.write("=== KẾT QUẢ TEST ===\n")
        f.write(f"R2 Score: {results['r2_score']:.4f}\n")
        f.write(f"MAE: {results['mae']:,.0f}\n\n")
        f.write(f"CV Mean R2 (Training): {results['cv_mean_r2']:.4f}\n\n")
        
        f.write("=== BỘ THAM SỐ TỐI ƯU ===\n")
        for param, value in results["best_params"].items():
            f.write(f"{param}: {value}\n")

    print(f"✓ Thông tin chi tiết lưu tại: {info_path}")

    return model_path

# Chuẩn bị dữ liệu để lưu
results_to_save = {
    'r2_score': float(r2_tuned),
    'mae': float(mae_tuned),
    'cv_mean_r2': float(search.best_score_),
    'best_params': search.best_params_
}

print("\n--- ĐANG LƯU MÔ HÌNH ---")
save_model(best_model, preprocessor, results_to_save)
print("\nHoàn tất chương trình!")
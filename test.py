import pandas as pd
import joblib
import os
import glob
import sys

# ======================================================================
# 0. ĐỊNH NGHĨA HÀM PHỤ TRỢ (BẮT BUỘC ĐỂ LOAD ĐƯỢC MODEL)
# ======================================================================
def convert_to_string(df):
    """
    Hàm này được dùng trong FunctionTransformer lúc train.
    Joblib cần tìm thấy hàm này trong namespace hiện tại để deserialize model.
    """
    return df.astype(str)

# ======================================================================
# 1. HÀM TÌM VÀ TẢI MÔ HÌNH
# ======================================================================
def load_latest_model(model_dir='models'):
    """
    Tìm và tải file model (.pkl) mới nhất trong thư mục models.
    """
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Thư mục '{model_dir}' không tồn tại. Bạn đã chạy file training chưa?")

    # Tìm tất cả các file bắt đầu bằng lightgbm_model_ và kết thúc bằng .pkl
    search_pattern = os.path.join(model_dir, "lightgbm_model_*.pkl")
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file model nào trong thư mục '{model_dir}'")
    
    # Lấy file có thời gian sửa đổi gần nhất
    latest_file = max(files, key=os.path.getctime)
    print(f"-> Đang tải mô hình từ: {latest_file}")
    
    try:
        model = joblib.load(latest_file)
        print("✓ Tải mô hình thành công!")
        return model
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG] Không thể đọc file mô hình: {latest_file}")
        print(f"Chi tiết lỗi: {e}")
        print("Gợi ý: Kiểm tra xem phiên bản scikit-learn/lightgbm lúc train và lúc test có giống nhau không.")
        return None

# ======================================================================
# 2. HÀM TẠO DỮ LIỆU MẪU (MOCK DATA)
# ======================================================================
def create_sample_data():
    """
    Tạo một DataFrame chứa các mẫu công việc giả lập để test.
    Cấu trúc cột phải khớp với 'feature_columns' lúc train.
    """
    data = [
        {
            'company_name': 'Google',
            'title': 'Senior AI Engineer',
            'description': 'We are looking for an experienced AI engineer to lead our deep learning team.',
            'location': 'Mountain View, CA',
            'views': 1500,
            'formatted_work_type': 'Full-time',
            'remote_allowed': '1',
            'formatted_experience_level': 'Senior Level',
            'skills_desc': 'Python, TensorFlow, PyTorch, Leadership, PhD in Computer Science',
            'sponsored': 1,
            'application_type': 'Complex'
        },
        {
            'company_name': 'Local Startup',
            'title': 'Junior Web Developer',
            'description': 'Maintain website and fix bugs using HTML/CSS.',
            'location': 'Hanoi, Vietnam',
            'views': 50,
            'formatted_work_type': 'Contract',
            'remote_allowed': '0',
            'formatted_experience_level': 'Entry Level',
            'skills_desc': 'HTML, CSS, Basic JavaScript',
            'sponsored': 0,
            'application_type': 'Simple'
        },
        {
            'company_name': 'Unknown Corp',
            'title': 'Data Analyst',
            'description': None,
            'location': 'Remote',
            'views': None,
            'formatted_work_type': 'Part-time',
            'remote_allowed': '1',
            'formatted_experience_level': 'Mid-Senior Level',
            'skills_desc': 'Excel, SQL',
            'sponsored': 0,
            'application_type': None
        }
    ]
    return pd.DataFrame(data)

# ======================================================================
# 3. HÀM TIỀN XỬ LÝ (QUAN TRỌNG)
# ======================================================================
def preprocess_input(df):
    """
    Feature engineering thủ công trước khi đưa vào Pipeline.
    """
    df_processed = df.copy()
    
    if 'description' in df_processed.columns:
        df_processed['description'] = df_processed['description'].fillna("")
    
    if 'skills_desc' in df_processed.columns:
        df_processed['skills_desc'] = df_processed['skills_desc'].fillna("")
    
    print("-> Đang tạo cột 'all_text' từ description và skills_desc...")
    df_processed['all_text'] = df_processed['description'] + " " + df_processed['skills_desc']
    
    return df_processed

# ======================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ======================================================================
if __name__ == "__main__":
    print("--- BẮT ĐẦU TEST MÔ HÌNH ---")
    
    # 1. Tải model
    try:
        pipeline_model = load_latest_model()
        
        # --- KHẮC PHỤC LỖI: KIỂM TRA NGAY NẾU MODEL LÀ NONE ---
        if pipeline_model is None:
            print("\n[DỪNG CHƯƠNG TRÌNH] Do không tải được mô hình.")
            sys.exit(1) # Thoát chương trình ngay lập tức
            
    except FileNotFoundError as e:
        print(f"\n[LỖI FILE] {e}")
        sys.exit(1)

    # 2. Tạo dữ liệu mẫu
    print("\n-> Đang tạo dữ liệu mẫu...")
    df_sample = create_sample_data()
    print(df_sample[['title', 'location', 'formatted_experience_level']])

    # 3. Xử lý dữ liệu
    df_ready = preprocess_input(df_sample)

    # 4. Dự đoán
    print("\n-> Đang thực hiện dự đoán lương...")
    try:
        predictions = pipeline_model.predict(df_ready)
        
        # 5. Hiển thị kết quả
        print("\n" + "="*40)
        print(" KẾT QUẢ DỰ ĐOÁN LƯƠNG (Normalized Salary)")
        print("="*40)
        
        df_sample['Predicted_Salary'] = predictions
        
        for i, row in df_sample.iterrows():
            salary_fmt = f"${row['Predicted_Salary']:,.2f}"
            print(f"Job {i+1}: {row['title']}")
            print(f"  - Cấp bậc: {row['formatted_experience_level']}")
            print(f"  - Kỹ năng: {row['skills_desc'][:50]}...")
            print(f"  => LƯƠNG DỰ ĐOÁN: {salary_fmt}")
            print("-" * 40)
            
    except Exception as e:
        print(f"\n[LỖI KHI DỰ ĐOÁN]: {e}")
        print("Gợi ý: Kiểm tra lại các cột dữ liệu đầu vào có khớp với lúc train không.")
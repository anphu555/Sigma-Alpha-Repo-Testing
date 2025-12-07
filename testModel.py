# Ví dụ cách dùng (Inference)
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import ast
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Chỉ hiện lỗi nghiêm trọng, ẩn thông báo GPU
# import tensorflow as tf... (các dòng import khác ở dưới)

# 1. Load Resources
loaded_model = tf.keras.models.load_model('two_tower_model.keras')
loaded_mlb = pickle.load(open('mlb_vocab.pkl', 'rb'))
places_clean = pd.read_csv('places_clean.csv') # Load dữ liệu địa điểm để hiển thị tên

# Parse tags string về list
places_clean['tags'] = places_clean['tags'].apply(ast.literal_eval)
# Pre-calculate item vectors (làm 1 lần khi khởi động app)
item_vectors = loaded_mlb.transform(places_clean['tags'])

def get_recommendations(user_tags_list):
    # 2. Chuyển đổi input của người dùng thành vector
    # Lưu ý: user_tags_list ví dụ ['Nature', 'Beach']
    user_vector = loaded_mlb.transform([user_tags_list])
    
    # 3. Repeat user vector để khớp với số lượng item
    user_vectors_repeated = np.repeat(user_vector, len(item_vectors), axis=0)
    
    # 4. Dự đoán
    raw_scores = loaded_model.predict([user_vectors_repeated, item_vectors], verbose=0).flatten()
    
    # --- THÊM BƯỚC NÀY: QUY ĐỔI VỀ 1-5 SAO ---
    star_scores = raw_scores * 4.0 + 1.0
    
    # Lấy Top 5
    top_indices = star_scores.argsort()[-5:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "name": places_clean.iloc[idx]['name'],
            "score": round(float(star_scores[idx]), 2), # Làm tròn 2 số lẻ
            "raw_match": f"{raw_scores[idx]*100:.1f}%", # Thêm cột % độ hợp
            "tags": places_clean.iloc[idx]['tags']
        })
    return results

# Test thử
recommendations = get_recommendations(['Nature', 'Cave'])
for rec in recommendations:
    print(f"\nTên: {rec['name']}")
    print(f"Điểm: {rec['score']} sao")
    print(f"Độ khớp: {rec['raw_match']}")
    print(f"Tags: {', '.join(rec['tags'])}")
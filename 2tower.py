import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# ======================================================
# 1. LOAD & PREPROCESS DATA
# ======================================================
print("--- 1. Loading & Processing Data ---")

# Load Places
places_df = pd.read_csv('vietnam_tourism_data_200tags_with_province.csv')
# Load Ratings (File mới generate ở bước trước)
ratings_df = pd.read_csv('ratings_generated.csv')

# Hàm xử lý tags
def parse_and_split_tags(tag_str):
    try:
        tags = ast.literal_eval(tag_str)
        if not tags: return "Unknown", []
        province = tags[0]
        features = tags[1:] if len(tags) > 1 else []
        return province, features
    except:
        return "Unknown", []

# Áp dụng tách Tỉnh và Features
places_df[['province', 'clean_tags']] = places_df['tags'].apply(
    lambda x: pd.Series(parse_and_split_tags(x))
)

# --- A. XỬ LÝ PROVINCE (Cho Item Tower) ---
# Tạo từ điển map Tỉnh -> Số nguyên (ID)
province_list = places_df['province'].unique().tolist()
province2idx = {p: i for i, p in enumerate(province_list)}
num_provinces = len(province_list)
print(f"Số lượng tỉnh thành: {num_provinces}")

places_df['province_idx'] = places_df['province'].map(province2idx)

# --- B. XỬ LÝ TAGS (Cho cả User & Item Tower) ---
mlb = MultiLabelBinarizer()
# Fit trên tất cả các tags có trong dữ liệu (của địa điểm)
mlb.fit(places_df['clean_tags'])
num_tag_features = len(mlb.classes_)
print(f"Số lượng Tag features: {num_tag_features}")

# Convert Item Tags sang vector one-hot
item_tags_matrix = mlb.transform(places_df['clean_tags'])
# Tạo map: PlaceID -> (TagVector, ProvinceID)
place_id_to_features = {
    row['id']: (item_tags_matrix[i], row['province_idx']) 
    for i, row in places_df.iterrows()
}

# --- C. CHUẨN BỊ DỮ LIỆU TRAIN ---
# Xử lý User Prefs (từ string -> list -> vector)
ratings_df['user_prefs_list'] = ratings_df['user_prefs'].apply(lambda x: ast.literal_eval(x))
user_prefs_matrix = mlb.transform(ratings_df['user_prefs_list'])

# Tạo các mảng dữ liệu để đưa vào model
X_user = []           # Input cho User Tower
X_item_tags = []      # Input 1 cho Item Tower
X_item_prov = []      # Input 2 cho Item Tower
y_rating = []         # Label (Rating đã normalize hoặc để 0-1)

max_rating = 5.0

print("--- Preparing Training Arrays ---")
for i, row in ratings_df.iterrows():
    # User Input
    X_user.append(user_prefs_matrix[i])
    
    # Item Input
    p_id = row['place_id']
    if p_id in place_id_to_features:
        p_tags_vec, p_prov_id = place_id_to_features[p_id]
        X_item_tags.append(p_tags_vec)
        X_item_prov.append(p_prov_id)
        
        # Normalize rating về 0-1 cho Sigmoid output (hoặc giữ nguyên nếu dùng linear)
        y_rating.append(row['rating'] / max_rating) 

X_user = np.array(X_user)
X_item_tags = np.array(X_item_tags)
X_item_prov = np.array(X_item_prov)
y_rating = np.array(y_rating)

# Split Train/Val
X_u_train, X_u_val, X_it_train, X_it_val, X_ip_train, X_ip_val, y_train, y_val = train_test_split(
    X_user, X_item_tags, X_item_prov, y_rating, test_size=0.2, random_state=42
)

# ======================================================
# 2. XÂY DỰNG MODEL (TWO-TOWER VỚI EMBEDDING)
# ======================================================
print("--- 2. Building Model ---")

embedding_dim = 32 # Kích thước vector cuối cùng của User và Item

# --- USER TOWER ---
# Input là vector sở thích (Binary vector của Tags)
user_input = Input(shape=(num_tag_features,), name='user_input')
u = layers.Dense(128, activation='relu')(user_input)
u = layers.Dropout(0.2)(u)
u = layers.Dense(64, activation='relu')(u)
u = layers.Dense(embedding_dim, activation='linear', name='user_embedding')(u)

# --- ITEM TOWER ---
# Input 1: Tags của địa điểm
item_tag_input = Input(shape=(num_tag_features,), name='item_tags_input')
i_t = layers.Dense(128, activation='relu')(item_tag_input)
i_t = layers.Dense(64, activation='relu')(i_t)

# Input 2: Province ID (New Feature!)
item_prov_input = Input(shape=(1,), name='item_province_input')
# Embedding layer cho tỉnh: Học biểu diễn vector cho từng tỉnh
# input_dim = số lượng tỉnh, output_dim = vector kích thước 16
i_p = layers.Embedding(input_dim=num_provinces + 1, output_dim=16)(item_prov_input)
i_p = layers.Flatten()(i_p)

# Kết hợp Tags và Province
combined_item = layers.Concatenate()([i_t, i_p])
i = layers.Dense(64, activation='relu')(combined_item)
i = layers.Dense(embedding_dim, activation='linear', name='item_embedding')(i)

# --- MERGE & OUTPUT ---
# Tính độ tương đồng (Dot product)
dot_product = layers.Dot(axes=1, normalize=True)([u, i])
output = layers.Dense(1, activation='sigmoid')(dot_product)

# Compile
model = Model(inputs=[user_input, item_tag_input, item_prov_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ======================================================
# 3. TRAINING
# ======================================================
print("--- 3. Training ---")
history = model.fit(
    x={
        'user_input': X_u_train,
        'item_tags_input': X_it_train,
        'item_province_input': X_ip_train
    },
    y=y_train,
    batch_size=64,
    epochs=10,
    validation_data=(
        {
            'user_input': X_u_val,
            'item_tags_input': X_it_val,
            'item_province_input': X_ip_val
        }, 
        y_val
    )
)

# ======================================================
# 4. TEST GỢI Ý (INFERENCE)
# ======================================================
print("\n--- 4. Demo Suggestion ---")

# Lấy thử 1 user từ tập validation
sample_user_idx = 0
user_pref_vec = X_u_val[sample_user_idx:sample_user_idx+1]
real_rating = y_val[sample_user_idx] * 5.0

# Lấy thông tin item tương ứng để hiển thị
item_tag_vec = X_it_val[sample_user_idx:sample_user_idx+1]
item_prov_val = X_ip_val[sample_user_idx:sample_user_idx+1]

# Predict
pred_score = model.predict({
    'user_input': user_pref_vec,
    'item_tags_input': item_tag_vec,
    'item_province_input': item_prov_val
})[0][0]

user_tags = mlb.inverse_transform(user_pref_vec)[0]

print(f"--- THÔNG TIN USER DEMO ---")
print(f"User ID: {sample_user_idx}")
print(f"Sở thích (Tags): {user_tags}")
# Nếu bạn muốn xem user này thích tỉnh nào (dựa trên lịch sử rating cao)
# Lấy lịch sử rating 5 sao của user này trong tập dữ liệu gốc
history_high_rate = ratings_df[
    (ratings_df['user_id'] == ratings_df.iloc[sample_user_idx]['user_id']) & 
    (ratings_df['rating'] >= 4)
]
# Join với bảng places để lấy tên tỉnh
history_places = history_high_rate.merge(places_df, left_on='place_id', right_on='id')
top_provinces = history_places['province'].value_counts().head(3).index.tolist()
print(f"Các tỉnh hay đi (Dựa trên lịch sử): {top_provinces}")
print("-" * 30)


print(f"Dự đoán rating: {pred_score * 5.0:.2f} / 5.0")
print(f"Rating thực tế: {real_rating:.2f} / 5.0")

# --- HÀM TÌM KIẾM ĐỊA ĐIỂM PHÙ HỢP NHẤT CHO USER NÀY ---
# (Quét qua toàn bộ địa điểm trong Database)
print("\nTop 5 địa điểm gợi ý cho User này:")

# Chuẩn bị input cho toàn bộ places
all_place_tags = item_tags_matrix
all_place_provs = places_df['province_idx'].values

# Lặp lại vector user cho bằng số lượng places
user_vec_repeated = np.repeat(user_pref_vec, len(places_df), axis=0)

# Dự đoán hàng loạt
predictions = model.predict({
    'user_input': user_vec_repeated,
    'item_tags_input': all_place_tags,
    'item_province_input': all_place_provs
}, batch_size=256)

# Lấy top 5
top_indices = predictions.flatten().argsort()[-5:][::-1]

for idx in top_indices:
    place = places_df.iloc[idx]
    score = predictions[idx][0] * 5.0
    print(f"- [{place['province']}] {place['name']} (Score: {score:.2f})")
    # print(f"  Tags: {place['clean_tags']}")



print("--- ĐANG LƯU HỆ THỐNG ---")

# 1. Lưu Model (Bộ não AI)
# Format .keras là chuẩn mới, nhẹ và an toàn hơn .h5
model.save('two_tower_model.keras')
print("✅ 1. Đã lưu model: two_tower_model.keras")

# 2. Lưu bộ từ điển Tags (MultiLabelBinarizer)
# Để dịch: ['Nature', 'Hills'] -> [0, 1, 0, ...]
with open('mlb_vocab.pkl', 'wb') as f:
    pickle.dump(mlb, f)
print("✅ 2. Đã lưu từ điển Tags: mlb_vocab.pkl")

# 3. Lưu bộ từ điển Tỉnh (Province Mapping) - QUAN TRỌNG
# Để dịch: "Ha Noi" -> 15
with open('province_map.pkl', 'wb') as f:
    pickle.dump(province2idx, f)
print("✅ 3. Đã lưu từ điển Tỉnh: province_map.pkl")

# Kiểm tra file
print("\nDanh sách file trong thư mục:")
print(os.listdir('.'))
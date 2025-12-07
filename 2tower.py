import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from collections import Counter

# ==========================================
# 1. LOAD & CLEAN DỮ LIỆU ĐỊA ĐIỂM
# ==========================================
places_df = pd.read_csv('/kaggle/input/place-with-tags-full/vietnam_tourism_data_200tags_with_province.csv')

def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

places_df['tags'] = places_df['tags'].apply(parse_list)

# Lọc bỏ tag rác (xuất hiện < 5 lần)
all_tags = [tag for tags in places_df['tags'] for tag in tags]
tag_counts = Counter(all_tags)
valid_tags = {tag for tag, count in tag_counts.items() if count >= 5} # Chỉ giữ tag phổ biến

def filter_tags(tags):
    return [t for t in tags if t in valid_tags]

places_df['tags'] = places_df['tags'].apply(filter_tags)
# Bỏ các địa điểm mất hết tag sau khi lọc
places_df = places_df[places_df['tags'].apply(len) > 0].reset_index(drop=True)

print(f"Số lượng địa điểm sau khi lọc: {len(places_df)}")
print(f"Số lượng tag hợp lệ: {len(valid_tags)}")

# ==========================================
# 2. LOAD & CLEAN DỮ LIỆU USER (FIX LỖI MẤT DATA)
# ==========================================
ratings_df = pd.read_csv('/kaggle/input/balanced-rating/synthetic_ratings_balanced.csv')
ratings_df['user_prefs'] = ratings_df['user_prefs'].apply(parse_list)

# --- BƯỚC QUAN TRỌNG NHẤT: ĐỒNG BỘ HÓA TAG ---
# Chỉ giữ lại trong user_prefs những tag nào nằm trong valid_tags của Place
ratings_df['user_prefs_filtered'] = ratings_df['user_prefs'].apply(filter_tags)

# Loại bỏ những User mà sau khi lọc không còn tag nào (Tránh lỗi vector rỗng)
ratings_df = ratings_df[ratings_df['user_prefs_filtered'].apply(len) > 0]

# Tạo lại nhãn Rating (1-5 -> 0-1)
# Dùng qcut để cân bằng lại phân phối nếu cần (như bài trước)
ratings_df['rating_qcut'] = pd.qcut(ratings_df['rating'], q=5, labels=[1, 2, 3, 4, 5])
labels = (ratings_df['rating_qcut'].astype(int) - 1) / 4.0

print(f"Số lượng mẫu huấn luyện hợp lệ: {len(ratings_df)}")

# ==========================================
# 3. TẠO VECTOR ĐẶC TRƯNG (ONE-HOT ENCODING)
# ==========================================
mlb = MultiLabelBinarizer()
mlb.fit(places_df['tags']) # Học từ điển từ Places

# Transform cả 2 bên theo cùng 1 bộ từ điển
item_tags_matrix = mlb.transform(places_df['tags'])
user_tags_matrix = mlb.transform(ratings_df['user_prefs_filtered']) # Dùng cột đã filter

# Map Place ID sang Index
place_id_to_index = {pid: i for i, pid in enumerate(places_df['id'])}
place_ids = ratings_df['place_id'].values

# Tạo mảng input cho Item (dựa trên place_id trong bảng rating)
# Cần lọc rating nào có place_id hợp lệ (phòng trường hợp place bị xóa ở bước 1)
valid_place_ids = set(places_df['id'])
mask = [pid in valid_place_ids for pid in place_ids]

# Áp dụng mask lọc cuối cùng
X_user_features = user_tags_matrix[mask]
place_ids_final = place_ids[mask]
labels_final = labels[mask]

X_item_features = np.array([item_tags_matrix[place_id_to_index[pid]] for pid in place_ids_final])

# Chia Train/Test
X_u_train, X_u_val, X_i_train, X_i_val, y_train, y_val = train_test_split(
    X_user_features, X_item_features, labels_final, 
    test_size=0.2, random_state=42, stratify=labels_final
)

# ==========================================
# 4. TRAINING MODEL (DOT PRODUCT)
# ==========================================
embedding_dim = 32
num_features = X_user_features.shape[1]

# User Tower
user_input = Input(shape=(num_features,), name='user_input')
u = layers.Dense(128, activation='relu')(user_input)
u = layers.Dropout(0.2)(u)
u = layers.Dense(64, activation='relu')(u)
u = layers.Dense(embedding_dim, activation='linear')(u)

# Item Tower
item_input = Input(shape=(num_features,), name='item_input')
i = layers.Dense(128, activation='relu')(item_input)
i = layers.Dropout(0.2)(i)
i = layers.Dense(64, activation='relu')(i)
i = layers.Dense(embedding_dim, activation='linear')(i)

# Dot Product (Tính độ tương đồng Cosine)
dot = layers.Dot(axes=1, normalize=True)([u, i])
output = layers.Dense(1, activation='sigmoid')(dot)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    [X_u_train, X_i_train], y_train,
    batch_size=64,
    epochs=15,
    validation_data=([X_u_val, X_i_val], y_val),
    verbose=1
)

# ==========================================
# 5. DEMO KẾT QUẢ
# ==========================================
print("\n--- DEMO GỢI Ý ---")
# Lấy 1 user mẫu từ tập Val
idx = 0
sample_user_vec = X_u_val[idx].reshape(1, -1)
# In ra sở thích thật sự (đã được lọc) của user này
print(f"User Prefs: {mlb.inverse_transform(sample_user_vec)}")

# Dự đoán với tất cả địa điểm
all_items = item_tags_matrix
user_repeated = np.repeat(sample_user_vec, len(all_items), axis=0)
preds = model.predict([user_repeated, all_items]).flatten()
preds_scaled = preds * 4.0 + 1.0

# Top 5
top_idx = preds_scaled.argsort()[-5:][::-1]
for i in top_idx:
    print(f"[{preds_scaled[i]:.2f}] {places_df.iloc[i]['name']}")
    print(f"  Tags: {places_df.iloc[i]['tags'][:5]}")
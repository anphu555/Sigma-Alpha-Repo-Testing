import pandas as pd
import numpy as np
import ast
import random

# ======================================================
# PHẦN 1: XỬ LÝ DỮ LIỆU ĐỊA ĐIỂM (TÁCH TỈNH)
# ======================================================

# 1. Load dữ liệu gốc
places_df = pd.read_csv('vietnam_tourism_data_200tags_with_province.csv')

# 2. Hàm để parse string tags thành list
def parse_tags(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

places_df['tags_list'] = places_df['tags'].apply(parse_tags)

# 3. Tách Tỉnh (Tag đầu tiên) và Các Tag còn lại
# Nếu list rỗng thì để Unknown, ngược lại lấy phần tử đầu làm Tỉnh
places_df['province'] = places_df['tags_list'].apply(lambda x: x[0] if len(x) > 0 else "Unknown")
# Các tag còn lại (từ phần tử thứ 2 trở đi)
places_df['feature_tags'] = places_df['tags_list'].apply(lambda x: x[1:] if len(x) > 1 else [])

# Lưu lại file places đã xử lý (nếu cần dùng sau này)
places_df.to_csv('places_processed.csv', index=False)
print("Đã tách xong cột Province.")
print(places_df[['id', 'province', 'feature_tags']].head())

# ======================================================
# PHẦN 2: GENERATE RATINGS CÓ QUY LUẬT (USER PREFERENCES)
# ======================================================

# Lấy danh sách tất cả các tag và tỉnh để random
all_tags = [tag for tags in places_df['feature_tags'] for tag in tags]
unique_tags = list(set(all_tags))
unique_provinces = places_df['province'].unique().tolist()

# Cấu hình sinh dữ liệu
NUM_USERS = 2000       # Số lượng user giả lập
MIN_REVIEWS = 5        # Số review tối thiểu mỗi user
MAX_REVIEWS = 25       # Số review tối đa mỗi user

user_data = []

print(f"\nĐang sinh rating cho {NUM_USERS} users...")

for user_id in range(NUM_USERS):
    # --- TẠO USER PERSONA (GU NGƯỜI DÙNG) ---
    # 1. User này thích 1-3 tỉnh thành nào đó (Ví dụ: Quê quán + Nơi thích du lịch)
    fav_provinces = random.sample(unique_provinces, k=random.randint(1, 3))
    
    # 2. User này thích 3-5 loại hình du lịch (Ví dụ: Thích Biển, Thích Chùa chiền)
    fav_tags = random.sample(unique_tags, k=random.randint(3, 5))
    
    # --- CHỌN ĐỊA ĐIỂM ĐỂ RATE ---
    # Giả định: User thường đi du lịch ở những tỉnh họ thích (70%) 
    # và đi khám phá ngẫu nhiên nơi khác (30%)
    
    num_ratings = random.randint(MIN_REVIEWS, MAX_REVIEWS)
    
    # Lấy danh sách ID các địa điểm thuộc tỉnh User thích
    fav_prov_places_idx = places_df[places_df['province'].isin(fav_provinces)].index.tolist()
    # Lấy danh sách ID các địa điểm còn lại
    other_places_idx = places_df[~places_df['province'].isin(fav_provinces)].index.tolist()
    
    # Chọn địa điểm
    if not fav_prov_places_idx: # Trường hợp hiếm nếu tỉnh đó ko có địa điểm nào
        chosen_indices = random.sample(other_places_idx, k=min(len(other_places_idx), num_ratings))
    else:
        n_fav = int(num_ratings * 0.7) # 70% địa điểm thuộc tỉnh yêu thích
        n_other = num_ratings - n_fav
        
        picked_fav = random.sample(fav_prov_places_idx, k=min(len(fav_prov_places_idx), n_fav))
        picked_other = random.sample(other_places_idx, k=min(len(other_places_idx), n_other))
        chosen_indices = picked_fav + picked_other
        
    # --- TÍNH ĐIỂM RATING ---
    for idx in chosen_indices:
        place = places_df.iloc[idx]
        
        # --- LOGIC CŨ (BỎ) ---
        # score = 3.0 + np.random.normal(0, 0.5) 
        # if place['province'] in fav_provinces: score += 1.0
        # matches = set(fav_tags).intersection(set(place['feature_tags']))
        # score += len(matches) * 0.5
        
        # --- LOGIC MỚI (MẠNH HƠN) ---
        matches = set(fav_tags).intersection(set(place['feature_tags']))
        is_prov_match = place['province'] in fav_provinces
        
        if len(matches) > 0:
            # Nếu trúng Sở thích -> Điểm cực cao (4.5 - 5.5)
            # Cộng thêm điểm nếu trúng cả Tỉnh
            score = 4.5 + (0.5 if is_prov_match else 0) + np.random.normal(0, 0.2)
        elif is_prov_match:
            # Nếu chỉ trúng Tỉnh mà ko trúng Sở thích -> Điểm khá (3.5 - 4.5)
            score = 3.5 + np.random.normal(0, 0.3)
        else:
            # Không trúng gì cả -> Điểm thấp (1.0 - 2.5)
            score = 2.0 + np.random.normal(0, 0.5)
            
        # Clip và làm tròn
        rating = int(np.clip(round(score), 1, 5))
        
        user_data.append({
            'user_id': user_id,
            'place_id': place['id'], # ID gốc của địa điểm
            'rating': rating,
            'user_prefs': str(fav_tags) # Lưu lại sở thích để dùng cho User Tower
        })

# Tạo DataFrame và Lưu file
ratings_gen_df = pd.DataFrame(user_data)
ratings_gen_df.to_csv('ratings_generated.csv', index=False)

print(f"Hoàn thành! Đã sinh {len(ratings_gen_df)} dòng rating.")
print(ratings_gen_df.head())
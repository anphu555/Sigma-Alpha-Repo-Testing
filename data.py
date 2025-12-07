import pandas as pd
import numpy as np
import random
import json
from tqdm import tqdm

# 1. Load dữ liệu địa điểm
places_df = pd.read_csv('vietnam_tourism_data_with_tags.csv')

def parse_tags(x):
    try:
        if isinstance(x, str): return json.loads(x.replace("'", '"'))
        return x
    except: return []

places_df['tags'] = places_df['tags'].apply(parse_tags)
all_tags = list(set([tag for tags in places_df['tags'] for tag in tags]))

# Tạo tag map để tìm nhanh địa điểm theo tag
tag_to_place_indices = {tag: [] for tag in all_tags}
for idx, tags in enumerate(places_df['tags']):
    for tag in tags:
        tag_to_place_indices[tag].append(idx)

# 2. Cấu hình sinh dữ liệu CÂN BẰNG
NUM_USERS = 1000
POSITIVES_PER_USER = 10  # Cố tình chọn 10 cái khớp
NEGATIVES_PER_USER = 10  # Cố tình chọn 10 cái không khớp
synthetic_data = []

print("Đang sinh dữ liệu cân bằng (Balanced Data)...")

for user_id in tqdm(range(NUM_USERS)):
    # Tạo sở thích User
    user_prefs = random.sample(all_tags, k=random.randint(3, 6))
    
    # --- A. TẠO MẪU POSITIVE (Rating cao) ---
    # Tìm các địa điểm có chứa ít nhất 1 tag trong sở thích user
    candidate_indices = set()
    for tag in user_prefs:
        candidate_indices.update(tag_to_place_indices.get(tag, []))
    
    candidate_indices = list(candidate_indices)
    
    # Nếu tìm được candidates, chọn ngẫu nhiên
    if candidate_indices:
        chosen_positives = random.sample(candidate_indices, k=min(len(candidate_indices), POSITIVES_PER_USER))
        
        for place_idx in chosen_positives:
            place = places_df.iloc[place_idx]
            # Tính độ khớp thực tế để gán rating
            intersection = len(set(user_prefs) & set(place['tags']))
            
            # Logic Rating cao (3.5 - 5.0)
            if intersection >= 2:
                rating = np.random.uniform(4.2, 5.0)
            else:
                rating = np.random.uniform(3.5, 4.2)
                
            synthetic_data.append({
                'user_id': user_id,
                'place_id': place['id'],
                'rating': rating,
                'user_prefs': json.dumps(user_prefs)
            })
            
    # --- B. TẠO MẪU NEGATIVE (Rating thấp) ---
    # Chọn ngẫu nhiên địa điểm (xác suất cao là không khớp)
    # Để chắc chắn, ta có thể check lại intersection
    while True:
        neg_candidates = places_df.sample(n=NEGATIVES_PER_USER)
        valid_negatives = []
        for _, place in neg_candidates.iterrows():
            intersection = len(set(user_prefs) & set(place['tags']))
            if intersection == 0: # Chỉ lấy cái hoàn toàn không liên quan
                valid_negatives.append(place)
        
        # Gán rating thấp (1.0 - 2.5) cho các mẫu này
        for place in valid_negatives:
            rating = np.random.uniform(1.0, 2.5)
            synthetic_data.append({
                'user_id': user_id,
                'place_id': place['id'],
                'rating': rating,
                'user_prefs': json.dumps(user_prefs)
            })
            
        if len(valid_negatives) > 0:
            break # Đã có đủ mẫu negative cho user này

# Lưu file
ratings_df = pd.DataFrame(synthetic_data)
ratings_df.to_csv('synthetic_ratings_balanced.csv', index=False)

print(f"Đã sinh {len(ratings_df)} dòng dữ liệu.")
print("Phân bố rating mới:")
print(ratings_df['rating'].hist())
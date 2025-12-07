import pandas as pd
import ast
from collections import Counter

# ==========================================
# 1. CẤU HÌNH
# ==========================================
MIN_TAG_FREQ = 5 # Ngưỡng xuất hiện tối thiểu của tag
INPUT_PLACES = 'vietnam_tourism_data_with_tags.csv'
INPUT_RATINGS = 'synthetic_ratings_balanced.csv'

OUTPUT_PLACES = 'places_clean.csv'
OUTPUT_RATINGS = 'ratings_clean.csv'

# ==========================================
# 2. XỬ LÝ PLACES (ĐỊA ĐIỂM)
# ==========================================
print("--- Đang xử lý Places ---")
places_df = pd.read_csv(INPUT_PLACES)

# Hàm parse list an toàn
def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

places_df['tags'] = places_df['tags'].apply(parse_list)

# Đếm tần suất và tìm valid_tags
all_tags = [tag for tags in places_df['tags'] for tag in tags]
tag_counts = Counter(all_tags)
valid_tags = {tag for tag, count in tag_counts.items() if count >= MIN_TAG_FREQ}

print(f"Tổng số tag ban đầu: {len(tag_counts)}")
print(f"Số tag hợp lệ (>= {MIN_TAG_FREQ}): {len(valid_tags)}")

# Lọc tag trong từng địa điểm
def filter_tags(tags):
    return [t for t in tags if t in valid_tags]

places_df['tags'] = places_df['tags'].apply(filter_tags)

# Xóa địa điểm không còn tag nào
places_df = places_df[places_df['tags'].apply(len) > 0].reset_index(drop=True)

# Lưu file Places sạch
# Convert list về string để lưu CSV không bị lỗi format
places_df['tags'] = places_df['tags'].apply(str) 
places_df.to_csv(OUTPUT_PLACES, index=False)
print(f"-> Đã lưu: {OUTPUT_PLACES} ({len(places_df)} địa điểm)")

# ==========================================
# 3. XỬ LÝ RATINGS (NGƯỜI DÙNG)
# ==========================================
print("\n--- Đang xử lý Ratings ---")
ratings_df = pd.read_csv(INPUT_RATINGS)
ratings_df['user_prefs'] = ratings_df['user_prefs'].apply(parse_list)

# Đồng bộ: Chỉ giữ lại tag nào có trong valid_tags của Place
# Nếu user thích tag lạ (không có trong Place) thì bỏ qua tag đó
ratings_df['user_prefs_filtered'] = ratings_df['user_prefs'].apply(filter_tags)

# Xóa User không còn tag nào sau khi lọc
before_len = len(ratings_df)
ratings_df = ratings_df[ratings_df['user_prefs_filtered'].apply(len) > 0]
print(f"Đã loại bỏ {before_len - len(ratings_df)} dòng rating do user tags không khớp.")

# Cân bằng lại điểm số (Quantile Binning)
# Chia đều điểm số thành 5 mức 1-5
ratings_df['rating_raw'] = ratings_df['rating'] # Giữ lại gốc nếu cần tham khảo
ratings_df['rating_processed'] = pd.qcut(ratings_df['rating'], q=5, labels=[1, 2, 3, 4, 5])

# Chuẩn bị để lưu
final_ratings = ratings_df[['user_id', 'place_id', 'rating_processed', 'user_prefs_filtered']].copy()
final_ratings.rename(columns={'rating_processed': 'rating', 'user_prefs_filtered': 'user_prefs'}, inplace=True)
final_ratings['user_prefs'] = final_ratings['user_prefs'].apply(str) # Convert list to string

# Chỉ giữ lại rating của các địa điểm CÒN TỒN TẠI trong places_clean
valid_place_ids = set(places_df['id'])
final_ratings = final_ratings[final_ratings['place_id'].isin(valid_place_ids)]

final_ratings.to_csv(OUTPUT_RATINGS, index=False)
print(f"-> Đã lưu: {OUTPUT_RATINGS} ({len(final_ratings)} dòng tương tác)")
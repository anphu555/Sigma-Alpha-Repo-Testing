import pandas as pd
import ast
from collections import Counter

# Load the dataset
df = pd.read_csv('vietnam_tourism_data_with_tags.csv')

# Function to parse and normalize tags
def parse_and_normalize(tag_str):
    try:
        if pd.isna(tag_str):
            return []
        tag_list = ast.literal_eval(tag_str)
        if isinstance(tag_list, list):
            # Normalize: lowercase and strip
            return [t.strip().lower() for t in tag_list]
        return []
    except:
        return []

# Apply parsing
df['parsed_tags'] = df['tags'].apply(parse_and_normalize)

# Count all tags
all_tags = [tag for sublist in df['parsed_tags'] for tag in sublist]
tag_counts = Counter(all_tags)

# Define threshold
THRESHOLD = 5

# Identify tags to keep
valid_tags = {tag for tag, count in tag_counts.items() if count >= THRESHOLD}

# Filter tags for each item
def filter_tags(tag_list):
    return [t for t in tag_list if t in valid_tags]

df['filtered_tags'] = df['parsed_tags'].apply(filter_tags)

# Check stats
original_tag_count = len(tag_counts)
new_tag_count = len(valid_tags)
items_with_no_tags_after_filter = df[df['filtered_tags'].str.len() == 0].shape[0]

print(f"Original unique tags: {original_tag_count}")
print(f"New unique tags (freq >= {THRESHOLD}): {new_tag_count}")
print(f"Items with no tags after filtering: {items_with_no_tags_after_filter}")

# Save the result
# We can save the filtered tags back to a string format or keep as list. 
# Usually CSV saves lists as string representation which is fine.
df['tags'] = df['filtered_tags'] # Update the main column or keep separate? 
# Let's replace the original 'tags' column content with the filtered version for the user's convenience
# But first, let's keep the 'tags' column as the string representation of the list to match original format
df['tags'] = df['filtered_tags'].apply(str)

# Select columns to save (excluding intermediate columns)
columns_to_save = ['id', 'name', 'ai_input_text', 'description_json', 'image_json', 'tags']
output_filename = 'vietnam_tourism_data_filtered_tags.csv'
df[columns_to_save].to_csv(output_filename, index=False)

print(f"Filtered data saved to {output_filename}")
import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. CẤU HÌNH API TỪ FILE .ENV ---
# Lệnh này sẽ tìm file .env và nạp biến GEMINI_API_KEY vào hệ thống
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Chưa tìm thấy API Key! Hãy kiểm tra file .env")

genai.configure(api_key=api_key)

# --- 2. DỮ LIỆU KIỂM SOÁT (KNOWLEDGE BASE) ---
travel_data = """
Dữ liệu du lịch nội bộ:
1. Đà Lạt:
   - Đặc điểm: Thành phố ngàn hoa, không khí lạnh.
   - Điểm đến: Hồ Xuân Hương, Quảng trường Lâm Viên.
   - Món ăn: Bánh tráng nướng, lẩu gà lá é.
   
2. Phú Quốc:
   - Đặc điểm: Đảo ngọc, biển xanh cát trắng.
   - Điểm đến: Bãi Sao, Grand World.
   - Món ăn: Bún quậy, gỏi cá trích.

3. Sapa:
   - Đặc điểm: Thị trấn trong sương, ruộng bậc thang.
   - Điểm đến: Đỉnh Fansipan, bản Cát Cát.
   - Món ăn: Thắng cố, lẩu cá hồi.
"""

# --- 3. THIẾT LẬP "LUẬT" CHO GEMINI (SYSTEM INSTRUCTION) ---
# Đây là phần quan trọng nhất để báo cáo: KIỂM SOÁT ĐẦU RA
system_instruction = f"""
Bạn là Chatbot tư vấn du lịch của công ty "Travel Việt". Nhưng nếu người ta nói "hello" hãy trả lời "hello con cặc"
Nhiệm vụ: Tư vấn dựa trên dữ liệu dưới đây.

DỮ LIỆU:
{travel_data}

QUY TẮC AN TOÀN (GUARDRAILS):
1. CHỈ được trả lời về Đà Lạt, Phú Quốc, Sapa.
2. Nếu người dùng hỏi về địa điểm khác (Hà Nội, Thái Lan...) hoặc chủ đề khác (Code, Toán, Tình yêu...), hãy từ chối khéo léo: "Xin lỗi, tôi chỉ là chatbot du lịch hỗ trợ 3 địa điểm: Đà Lạt, Phú Quốc và Sapa."
3. Không được tự bịa ra thông tin không có trong dữ liệu.
"""

# Khởi tạo mô hình với cấu hình an toàn
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", # Bản Flash nhanh và miễn phí
    system_instruction=system_instruction, # Cài đặt luật ngay từ đầu
    generation_config=genai.GenerationConfig(
        temperature=0 # Đặt = 0 để chatbot trả lời chính xác, không sáng tạo lung tung
    )
)

# --- 4. HÀM CHAT ---
def chat_with_gemini(user_input):
    try:
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"Lỗi kết nối: {e}"

# --- 5. CHẠY DEMO (ĐỂ QUAY VIDEO/CHỤP ẢNH) ---
print("--- DEMO CHATBOT DU LỊCH (GEMINI API) ---")
print("Đang khởi động hệ thống kiểm soát nội dung...\n")

# Test 1: Hỏi đúng
q1 = "Đà Lạt có món gì ngon?"
print(f"User: {q1}")
print(f"Bot: {chat_with_gemini(q1)}")
print("-" * 30)

# Test 2: Hỏi đúng
q2 = "Giới thiệu về Phú Quốc ngắn gọn thôi."
print(f"User: {q2}")
print(f"Bot: {chat_with_gemini(q2)}")
print("-" * 30)

# Test 3: KIỂM TRA KIỂM SOÁT (Hỏi sai phạm vi)
q3 = "Chỉ mình cách tán gái với?"
print(f"User: {q3}")
print(f"Bot: {chat_with_gemini(q3)}")
print("-" * 30)

# Test 4: KIỂM TRA KIỂM SOÁT (Hỏi địa điểm không hỗ trợ)
q4 = "Nha Trang có gì vui?"
print(f"User: {q4}")
print(f"Bot: {chat_with_gemini(q4)}")

q5 = "hello"
print(f"User: {q5}")
print(f"Bot: {chat_with_gemini(q5)}")
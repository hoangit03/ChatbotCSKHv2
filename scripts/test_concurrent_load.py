import asyncio
import httpx
import time
import os

# Cố gắng đọc DEV_API_KEY từ file .env nếu có
def get_api_key():
    api_key = "chatbot-ctlotus" # Giá trị mặc định thường dùng
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DEV_API_KEY="):
                    api_key = line.strip().split("=")[1].strip().strip("\"'")
                    break
    return api_key

API_URL = "http://localhost:8000/api/v1/chat"
API_KEY = get_api_key()
NUM_USERS = 30

async def send_request(client: httpx.AsyncClient, user_id: int):
    payload = {
        "message": f"Dự án này có tiện ích gì nổi bật không? (User {user_id})",
        "project_name": "Vinhomes",
        "session_id": f"test_load_user_{user_id}"
    }
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    try:
        response = await client.post(API_URL, json=payload, headers=headers, timeout=60.0)
        elapsed = time.time() - start_time
        if response.status_code == 200:
            data = response.json()
            print(f"[User {user_id:02d}] ✅ Success in {elapsed:.2f}s | Response: {data.get('answer', '')[:50]}...")
            return True, elapsed
        else:
            print(f"[User {user_id:02d}] ❌ Failed in {elapsed:.2f}s | Status: {response.status_code} | {response.text}")
            return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[User {user_id:02d}] ❌ Error in {elapsed:.2f}s | Exception: {str(e)}")
        return False, elapsed

async def main():
    print(f"Bắt đầu giả lập {NUM_USERS} users gửi request cùng lúc tới {API_URL}...")
    print(f"Sử dụng API Key: {API_KEY[:8]}...")
    
    # Dùng httpx.AsyncClient để tái sử dụng connection
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=100)
    async with httpx.AsyncClient(limits=limits) as client:
        start_total = time.time()
        
        # Tạo danh sách các task cần chạy đồng thời
        tasks = [send_request(client, i+1) for i in range(NUM_USERS)]
        
        # Chạy toàn bộ tasks cùng một lúc bằng asyncio.gather
        results = await asyncio.gather(*tasks)
        
        end_total = time.time()
        
    total_time = end_total - start_total
    success_count = sum(1 for success, _ in results if success)
    fail_count = NUM_USERS - success_count
    
    print("\n" + "="*50)
    print("🚀 BÁO CÁO KẾT QUẢ LOAD TEST")
    print("="*50)
    print(f"Tổng số request   : {NUM_USERS}")
    print(f"Thành công        : {success_count} ✅")
    print(f"Thất bại/Lỗi      : {fail_count} ❌")
    print(f"Tổng thời gian    : {total_time:.2f} giây")
    if success_count > 0:
        avg_time = sum(time for success, time in results if success) / success_count
        print(f"Thời gian TB/Req  : {avg_time:.2f} giây")
    print("="*50)

if __name__ == "__main__":
    # Khắc phục lỗi EventLoop của Windows nếu có
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())

"""Week 10: Concurrent API Tester."""

import requests
import concurrent.futures
import time

API_URL = "http://127.0.0.1:8000"

def test_health():
    """Test the health endpoint."""
    resp = requests.get(f"{API_URL}/health")
    assert resp.status_code == 200
    print(f"Health: {resp.json()}")

def test_metadata():
    """Test the metadata endpoint."""
    resp = requests.get(f"{API_URL}/metadata")
    assert resp.status_code == 200
    print(f"Metadata: {resp.json()}")

def make_query_request(worker_id: int):
    """Make a POST request to /query."""
    payload = {
        "question": f"What is the impact of deep learning? (Request {worker_id})",
        "k": 3
    }
    
    start = time.time()
    resp = requests.post(f"{API_URL}/query", json=payload)
    end = time.time()
    
    return {
        "worker_id": worker_id,
        "status_code": resp.status_code,
        "time": end - start,
        "response": resp.json() if resp.status_code == 200 else resp.text
    }

def run_concurrent_test(num_requests: int = 10):
    """Run concurrent requests to test API stability."""
    print(f"\nLaunching {num_requests} concurrent requests to /query ...")
    
    success_count = 0
    failure_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_query_request, i) for i in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res["status_code"] == 200:
                    success_count += 1
                    print(f"[Worker {res['worker_id']}] Success ({res['time']:.2f}s) - Answer: {res['response']['answer'][:50]}...")
                else:
                    failure_count += 1
                    print(f"[Worker {res['worker_id']}] Failed ({res['status_code']}): {res['response']}")
            except Exception as e:
                failure_count += 1
                print(f"Worker failed with exception: {e}")
                
    print(f"\n--- Load Test Results ---")
    print(f"Total Requests: {num_requests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")

if __name__ == "__main__":
    try:
        test_health()
        test_metadata()
        run_concurrent_test(5)  # Limit to 5 to avoid completely exhausting OpenRouter free credits
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is Uvicorn running on port 8000?")

import requests
import traceback

try:
    print("Testing /predict...")
    res1 = requests.post("http://localhost:8000/predict", json={"text": "I am so happy!"})
    print(res1.status_code)
    print(res1.json())
except Exception as e:
    print("Predict failed")
    traceback.print_exc()

try:
    print("Testing /explain...")
    res2 = requests.post("http://localhost:8000/explain", json={"text": "I am so happy!"})
    print(res2.status_code)
    print(res2.json())
except Exception as e:
    print("Explain failed")
    traceback.print_exc()

import requests
import json

# FastAPI endpoint
url = "http://10.67.18.2:8599/predict_multiple/"  # Update to your server's actual URL

# File to upload
file_path = "sample.png"  # Can also be an image file

# Questions and tags
questions_payload = [
    {"question": "What is the person account number ?", "tag": "dhsh"},
    {"question": "What is the person name ?", "tag": "dstha"}
]

# Convert to JSON string
questions_str = json.dumps(questions_payload)

# Form data
form_data = {
    "questions_str": questions_str,
    "model_type": "MGVG"
}

# API key in X-API-KEY header
headers = {
    "X-API-KEY": "AGRIBOT"  # Replace with your actual API key
}

# Prepare file payload
with open(file_path, "rb") as f:
    files = {
        "file": (file_path, f, "application/octet-stream")
    }

    # POST request
    response = requests.post(url, data=form_data, files=files, headers=headers)

# Print response
print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception:
    print("Non-JSON Response:", response.text)


[
        {
            "question":"What is the account number",
            "tag": "jkwjbkejbf"
        },
        {
            "question":"What is the name of the bank",
            "tag": "oiwhekjb"
        },
        {
            "question":"What is the IFSC number",
            "tag": "oiwhergerekjb"
        },
        { "question":"What is the address",
            "tag": "uiuejbirjb"
        },
        {
            "question":"What is type of account",
            "tag": "uwyijebkfjb"
        }
]
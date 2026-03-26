from fastapi import FastAPI, UploadFile
import requests

app = FastAPI()

API_TOKEN = "ВАШ_API_KEY"

@app.post("/process-image")
async def process_image(file: UploadFile):
    image_bytes = await file.read()
    
    # Отправляем на AI сервис (например Replicate)
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={"Authorization": f"Token {API_TOKEN}"},
        files={"file": image_bytes},
        json={"version": "lama-text-removal"}  # модель LaMa
    )

    result = response.json()
    return {"result": result}

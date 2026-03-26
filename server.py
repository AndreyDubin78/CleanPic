from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()
reader = easyocr.Reader(['en', 'ru'])  # английский и русский

@app.post("/remove_text")
async def remove_text(file: UploadFile = File(...)):
    # Чтение картинки
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # Распознаём текст
    result = reader.readtext(image)
    extracted_text = "\n".join([res[1] for res in result])

    # Маска текста
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Удаление текста (inpainting)
    image_no_text = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)

    # Сохраняем изображение
    output_img_path = "output.png"
    cv2.imwrite(output_img_path, cv2.cvtColor(image_no_text, cv2.COLOR_RGB2BGR))

    # Сохраняем текст
    output_text_path = "extracted_text.txt"
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    return {
        "image_file": output_img_path,
        "text_file": output_text_path,
        "text": extracted_text
    }

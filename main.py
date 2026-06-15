from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

app = FastAPI()


@app.get("/")
def root():
    return {"message": "FastAPI image upload server is running"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(BytesIO(contents))
    width, height = image.size

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "width": width,
        "height": height,
    }

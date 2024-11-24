from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import os
from fastapi.middleware.cors import CORSMiddleware
from face_searching.fr import live_face_srching_api as f_srch
from fastapi.responses import FileResponse

app = FastAPI()
origins = ["http://localhost", "http://localhost:5500", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))


@app.post("/upload")
async def upload_images(
    files: list[UploadFile] = File(...), textInputs: list = Form(...)
):
    # responses = []
    img_files = []
    for file in files:
        # Save the uploaded file in the current directory
        file_path = os.path.join(current_directory, "./input/", file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        img_files.append(f"D:/Rajasthan_Project/apis/input/{file.filename}")
    # print(textInputs)
    auth_id = ""
    filter_by = ""
    filter_data = ""
    for textInput in textInputs:
        print("Received text:", textInput)
    # print(img_files)

    # CALL API
    # img_files = [
        # "D:/Rajasthan_Project/apis/face_searching/fr/known_people/Sagar_Shukla.jpg",
        # "D:/Rajasthan_Project/apis/face_searching/fr/known_people/Chandresh.jpg",
        # "D:/Rajasthan_Project/apis/face_searching/fr/known_people/Marcus_Michael.jpg",
        # "D:/Rajasthan_Project/apis/face_searching/fr/known_people/Toufiq.png",
    # ]
    response = f_srch.initiate_face_srch("ABC123", "area", "814110", img_files)
    for i,r in enumerate(response):
        print(response[i])
        response[i]["img"] = FileResponse(r["img_name"])
    return response

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5666)
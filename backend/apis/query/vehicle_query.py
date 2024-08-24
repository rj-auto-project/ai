from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class vehicle_prop(BaseModel):
    vehicle_type:str
    color:str
    brand:str
    model:str
    license_no:str
    cam_urls:[str]

@app.post("/query/vehicle")
def query_vehicle(data:vehicle_prop):
    data = {**data.dict()}
    print(data)
    return data

if "__main__" == __name__:
    uvicorn.run(app, host="127.0.0.1", port=5667)


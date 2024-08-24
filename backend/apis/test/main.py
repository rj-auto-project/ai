import json
import os
from typing import Literal, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
import random,cv2,asyncio
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from mangum import Mangum
import mysql.connector
import uvicorn
camera = cv2.VideoCapture('rtmp://122.200.18.78/live/foobar')
class Book(BaseModel):
    name: str
    genre: Literal["fiction", "non-fiction"]
    price: float
    book_id: Optional[str] = uuid4().hex


BOOKS_FILE = "books.json"
BOOKS = []

if os.path.exists(BOOKS_FILE):
    with open(BOOKS_FILE, "r") as f:
        BOOKS = json.load(f)

app = FastAPI()
handler = Mangum(app)


@app.get("/")
def say_hello():
    return {"message": "Hello World"}


@app.get("/lelo/{name}")
async def say_hello_with_name(name: str):
    return {"message": f"Hello {name}"}


@app.get("/random-book")
async def random_book():
    return random.choice(BOOKS)


@app.get("/list-books")
async def list_books():
    return {"books": BOOKS}


@app.get("/book_by_index/{index}")
async def book_by_index(index: int):
    if index < len(BOOKS):
        return BOOKS[index]
    else:
        raise HTTPException(404, f"Book index {index} out of range ({len(BOOKS)}).")


@app.post("/add-book")
async def add_book(book: Book):
    book.book_id = uuid4().hex
    json_book = jsonable_encoder(book)
    BOOKS.append(json_book)

    with open(BOOKS_FILE, "w") as f:
        json.dump(BOOKS, f)

    return {"book_id": book.book_id}


@app.get("/get-book")
async def get_book(book_id: str):
    for book in BOOKS:
        if book.book_id == book_id:
            return book

    raise HTTPException(404, f"Book ID {book_id} not found in database.")

@app.websocket("/ws")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")

@app.get("/db")
async def fetch_db():
    response = []
    con = mysql.connector.connect(host="localhost", user="root", password="", database="xyz")
    cursor = con.cursor()
	# if data["areas"][0] == "*":
	# query2 = f"select * from dvr where auth_id = {data['auth_id']}"
    query2 = f"select * from dvr"
	# else:
	# 	query2 = f"select * from dvr where auth_id = {data['auth_id']} and pincode in {data['areas']}"
    cursor.execute(query2)

    table = cursor.fetchall()

    print('\n Table Data:')
    for row in table:
        r = {
                "DVR_loc":f"{row[0]}",
                "Associated Cameras":f"{row[1]}",
                "Owenership":f"{row[2]}",
                "owner_name": f"{row[4]}",
                "DVR_id":f"{row[5]}",
                "DVR_status":f"{row[6]}",
            }
        response.append(r)
    cursor.close()

    con.close()
    return response

if __name__ == '__main__':
    # uvicorn.run(app, host='0.0.0.0', port=8000)
    uvicorn.run(app, host='127.0.0.1', port=5666)
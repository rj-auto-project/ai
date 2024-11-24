









# import asyncio
# import websockets
# import json

# async def search_vehicle():
#     uri = "ws://localhost:5777/ws/search?vehicle=tractor&color=red&initial_timestamp=2024-06-19T20:45:24&final_timestamp=2024-06-20T23:45:24"
#     async with websockets.connect(uri) as websocket:
#         while True:
#             try:
#                 response = await websocket.recv()
#                 data = json.loads(response)
#                 if len(data)==0:
#                     print("Number plate not found, checking again...")
#                 elif list(data[0].keys())[0] == "end":
#                     print("search end")
#                     break
#                 else:
#                     print("Number plate found:", data)
#                     # break
#             except websockets.exceptions.ConnectionClosedError as e:
#                 print(f"Connection closed: {e}")
#                 break

# asyncio.run(search_vehicle())
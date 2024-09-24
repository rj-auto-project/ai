
# create d_temp folder and cc_temp in stream folder if not already created
# check if 10 processes are running in multiprocessing pool if not then check which process-id is not running and rerun it
# get coordinates from postgre table o table name camera  where rows names are RLV_line, NP_line, OS_line, WWD_line, all have a integer[] datatype, here RLV_line have 2 elements in array, NP_line have 1


# red-light violation = crossing line pair 1-2
# no parking - crossing line single line 1
# overspeediing - speed line pair line 1-2
# wrong way driving - entry exit pair - line 1


import os
import redis
from module import  process_raw_d_logs, process_d_logs, process_raw_cc_logs, process_cc_logs

r = redis.Redis(host='localhost', port=6379, db=0)


# Function to create directories if they don't already exist
def create_directories():
    base_dir = '/home/annone/ai-camera/backend/stream'
    d_temp = os.path.join(base_dir, 'd_temp')
    cc_temp = os.path.join(base_dir, 'cc_temp')

    os.makedirs(d_temp, exist_ok=True)
    os.makedirs(cc_temp, exist_ok=True)

    if os.listdir():
        True
    else:
        False
    print(f"Directories '{d_temp}' and '{cc_temp}' checked/created.")

def clean_redis_db():
    process_raw_d_logs()
    process_d_logs()
    process_raw_cc_logs
    process_cc_logs()


create_directories()
clean_redis_db()
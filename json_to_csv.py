import json
import os
import pandas as pd
import numpy as np

json_files = os.listdir("json")
os.makedirs("csv", exist_ok=True)
with open("example.csv","r") as f:
    csv_heads = f.readline().strip().split(",")

def process_json(file):
    try:
        with open("json/"+file, "r") as f:
            data = json.load(f)
    except:
        return []
    
    rows = []
    action_taken = False
    for frame in data:
        new_row = dict.fromkeys(csv_heads, " ")
        frame_id = frame["frame_id"]
        real_frame_id = frame_id * 30 + 1
        new_row["ID"] = file.replace(".json", "") + "_" + str(real_frame_id)
        if frame["action"]:
            action_taken = True
        
        if action_taken:
            new_row["Driver_State_Changed"] = "True"
        else:
            new_row["Driver_State_Changed"] = "False"
        danger_count = 0
        
        # sort bbox using bbox["id"]
        frame["bbox"].sort(key=lambda x: x["id"])
        
        
        for i in range(22):
            new_row[f"Hazard_Track_{i}"] = ""
        for bbox in frame["bbox"]:
            des = bbox["des"]
            is_danger = bbox["is_danger"]
            if is_danger:
                new_row[f"Hazard_Track_{danger_count}"] = str(int(bbox["id"]))
                new_row[f"Hazard_Name_{danger_count}"] = des
                danger_count += 1
            

        for id, tmp in enumerate([new_row.copy() for _ in range(30)]):
            tmp["ID"] = file.replace(".json", "") + "_" + str(real_frame_id + id)
            rows.append(tmp)
    print([i["ID"] for i in rows])
    return rows
    # pd.DataFrame(rows).to_csv("csv/"+file.replace(".json", ".csv"), index=False)

import random
all_rows = []
for file in json_files:
    all_rows += process_json(file)

pd.DataFrame(all_rows).to_csv("csv/"+file.replace(".json", ".csv"), index=False)

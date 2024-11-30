# %%
import pickle
with open("annotations.pkl", "rb") as f:
    annotations = pickle.load(f)
# %%
import os
import cv2
videos = os.listdir("COOOL Benchmark")
# video = videos[0]
colors = ["red", "blue", "yellow", "green", "purple", "orange", "pink", "brown", "black", "white", "gray", "lightblue", "lightgreen", "lightyellow", "lightred", "lightpurple", "lightorange", "lightpink", "lightbrown", "lightblack", "lightwhite", "lightgray"]
from matplotlib import colors as c

def color_name_to_rgb(color):
    return [i*255 for i in c.to_rgb(color)]

from openai import OpenAI
import base64
from PIL import Image
import io

def video_to_prompt(frames):
    res = []
    for frame in frames[::10]:
        frame_img = Image.fromarray(frame)
        buffer = io.BytesIO()
        frame_img.save(buffer, format="JPEG")
        frame_base64 = base64.b64encode(buffer.getvalue()).decode()


        res.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{frame_base64}","detail":"high"}})
    return res
    

prompt = f"""
You are a data labeler for self-driving cars. You will be shown a list of video frames, with multiple bounding boxes each indicating a potential object. 
Your task is to identify any instance where the autonomous driving system should press the break due to danger. 
For each dangerous object, record the bounding box number (indexed starting at 1) 
and a brief, 32-character description of the danger, including what is the object. Ignore objects that do not present a danger.

In addition, specify the type of action taken for the whole frame.

Bounding boxes are color-coded and can appear in these colors: {", ".join(colors)}. 
Return your response in JSON format as follows:

Do not be overlly cautious. Only report dangers that are clearly visible and present a real threat! DO **NOT** stop just because you see something on the road, it is NORMAL to see cars on the road, that is what road is for.

[
{{
    "frame_id": 0
    "whole_image_description": "description of the whole image",
    "bbox_description": [ //including all boxes here, even if they do not present a danger
        {{"bbox_id": 1, "bbox_color": "color", "description": "description of box", "is_danger": True/False}},
    ],
    "action_taken": True/False,
}}
,
...
"""

def generate_prompt(video):
    annotation = annotations[video.replace(".mp4", "")]
    cap = cv2.VideoCapture("COOOL Benchmark/" + video)
    frames = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = annotation[frame_id]["challenge_object"]
        for _, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(i) for i in box["bbox"]]
            color = colors[int(box["track_id"])]
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color_name_to_rgb(color), 2)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_id += 1

    client = OpenAI(api_key="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")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": 
            [
                {"type": "text", "text": prompt + f"You have {len(frames[::10])} frames"},
                *video_to_prompt(frames)
            ]},
        ],
        temperature=0.1,
        response_format={ "type": "json_object" }
    ) 


    # %%
    import json
    len(json.loads(completion.choices[0].message.content)), len(frames[::10])
 
    # %%
    with open("json/"+video.replace(".mp4", ".json"), "w") as f:
        f.write(completion.choices[0].message.content)


# use a pool of len(videos) workers to do the processing, use thread since it is not compute bound
from multiprocessing.pool import ThreadPool
pool = ThreadPool(len(videos))
pool.map(generate_prompt, videos)

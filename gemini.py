# %%
# colors = ["red", "blue", "yellow", "green", "purple", "orange", "pink", "brown", "black", "white", "gray", "lightblue", "lightgreen", "olive", "cyan", "gold", "peru", "darkred", "darkblue", "darkolivegreen", "yellowgreen", "lightcyan", "wheat", "tan", "lime" "slategrey"] 
from matplotlib import colors as c

colors =  list(c.CSS4_COLORS.keys())+list(c.CSS4_COLORS.keys())

prompt = f"""
You are a data labeler for self-driving cars. You will be shown a list of video frames and the optical flow, with multiple bounding boxes each indicating a potential object. 
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
    "des": "description of the whole image",
    "bbox": [ //including all boxes here, even if they do not present a danger
        {{ "color": "bbox_color", "id": 1, "des": "description of box", "is_danger": True/False}}, 
        // note that the bbox_id is not necessarily the same as the index of the box in the list, it is theindex of the color and the index of boxes since
        the first frame of video. For example, frame 1 could has bbox 1,2,3, and frame 2 could have 2, 3, 4
    ],
    "action": True/False,
}}
,
... for every single frames in the video
"""
import os
os.makedirs("json", exist_ok=True)
# %%
from dotenv import load_dotenv
import os
import os
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_AI_KEY'])

# %%
# %%
import pickle
with open("annotations.pkl", "rb") as f:
    annotations = pickle.load(f)
annotations["video_00013"] = annotations["video_0013"]
# %% 
import os
import cv2
videos = os.listdir("COOOL Benchmark")
# video = videos[0]
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
        frame_img = frame_img.resize((256, 256))
        buffer = io.BytesIO()
        frame_img.save(buffer, format="JPEG")
        buffer.seek(0)
        f = genai.upload_file(path=buffer, mime_type="image/jpeg")
        res.append(frame_img) 
        
    return res

# %%

def generate(video):
    print(video)
    if video.replace(".mp4", ".json") in os.listdir("json"):
        return
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
        
    cap = cv2.VideoCapture("flow/" + video + ".mp4")
    flow_frames = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flow_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_id += 1
        
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    text = model.generate_content(
                [prompt, *video_to_prompt(frames), *video_to_prompt(flow_frames)],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                ), 
            ).text
    with open(f"json/{video.replace('.mp4', '.json')}", "w") as f:
        f.write(text)

    

# %%
for video in videos:
    generate(video)


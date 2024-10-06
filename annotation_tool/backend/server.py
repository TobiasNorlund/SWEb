from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import subprocess
import json

app = FastAPI()

DATA_FILE = "data/data.jsonl"

with open(DATA_FILE, "r") as f:
    data = []
    for line in f:
        parsed = json.loads(line)

        # Add annotation fields
        if "selected_lines" not in parsed:
            parsed["selected_lines"] = []
        if "is_ignored" not in parsed:
            parsed["is_ignored"] = False
        
        data.append(parsed)


# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"hello": "world"}


@app.get("/example/{id}")
async def get_example_id(id: int):
    return {
        "id": id,
        "total_examples": len(data),
        "url": data[id]["url"],
        "content": data[id]["content"],
        "selected_lines": data[id]["selected_lines"],
        "is_ignored": data[id]["is_ignored"],
        "line_predictions": data[id]["line_predictions"] if "line_predictions" in data[id] else [],
    }


@app.post("/example/{id}")
async def save_annotation(id: int, annotation: dict):
    data[id]["selected_lines"] = annotation["selected_lines"]
    data[id]["is_ignored"] = annotation["is_ignored"]
    with open(DATA_FILE, "w") as f:
        for ex in data:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0")
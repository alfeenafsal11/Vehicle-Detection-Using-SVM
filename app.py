# app.py
import os
import uuid
import gradio as gr
from main import detect_vehicles_in_video  # adjust import if you used main.py

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_detection(video):
    # Gradio passes local temp file path
    if video is None:
        raise gr.Error("No video provided.")
    input_path = video
    out_name = f"processed_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    stats = detect_vehicles_in_video(input_path, out_path, resize_to=(640,360), frame_step=1)
    summary = f"Total vehicles detected: {stats.get('total_vehicles', 'N/A')}"
    return out_path, summary

demo = gr.Interface(
    fn=run_detection,
    inputs=gr.Video(label="Upload MP4 Video"),
    outputs=[gr.Video(label="Processed Video"), gr.Textbox(label="Summary")],
    title="Vehicle Detection – HOG+SVM",
    description="Upload a traffic video. Returns annotated video + vehicle count."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

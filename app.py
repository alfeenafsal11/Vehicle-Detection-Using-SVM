# app.py
import os
import uuid
import gradio as gr
from main import detect_vehicles_in_video

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_detection(video, resize_w=640, resize_h=360, step=8, scales=[1.0,0.75], threshold=0.0):
    if video is None:
        raise gr.Error("No video uploaded.")
    # Gradio passes a local temp file path for 'video'
    out_name = f"processed_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    resize = None if resize_w == 0 or resize_h == 0 else (resize_w, resize_h)
    stats = detect_vehicles_in_video(
        input_path=video,
        output_path=out_path,
        resize_to=resize,
        frame_step=1,
        step_size=step,
        scales=scales,
        min_score_threshold=threshold
    )
    summary = f"Total vehicles detected: {stats.get('total_vehicles', 'N/A')}"
    return out_path, summary

with gr.Blocks() as demo:
    gr.Markdown("# Vehicle Detection System — HOG + SVM (Demo)")
    with gr.Row():
        video_in = gr.Video(label="Upload MP4 video")
        with gr.Column():
            resize_w = gr.Number(value=640, label="Resize width (0 = original)")
            resize_h = gr.Number(value=360, label="Resize height (0 = original)")
            step = gr.Slider(4, 16, value=8, step=1, label="Sliding window step (px)")
            threshold = gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label="SVM threshold")
            scales_text = gr.Textbox(value="1.0,0.75", label="Scales (comma-separated)")
            run_btn = gr.Button("Run Detection")
    video_out = gr.Video(label="Processed video")
    text_out = gr.Textbox(label="Summary", interactive=False)

    def on_run(video, rw, rh, step_val, thresh, scales_text_val):
        scales = [float(s.strip()) for s in scales_text_val.split(",") if s.strip()]
        return run_detection(video, int(rw), int(rh), int(step_val), scales, float(thresh))

    run_btn.click(on_run, inputs=[video_in, resize_w, resize_h, step, threshold, scales_text], outputs=[video_out, text_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

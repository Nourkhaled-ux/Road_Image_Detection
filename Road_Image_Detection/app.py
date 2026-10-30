import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import time


MODEL_PATH = "yolo11s_best.pt"
model      = YOLO(MODEL_PATH)

CLASS_NAMES  = ['alligator','block','crack','edge','longitudinal','pothole','transverse']
CLASS_COLORS = {
    'alligator':    (255,  80,  80),
    'block':        ( 80, 160, 255),
    'crack':        ( 80, 255,  80),
    'edge':         (255, 255,  80),
    'longitudinal': (255, 165,   0),
    'pothole':      (180,  80, 255),
    'transverse':   ( 80, 255, 210),
}

SEVERITY = {
    'pothole':      'High',
    'alligator':    'High',
    'block':        'Medium',
    'longitudinal': 'Medium',
    'transverse':   'Medium',
    'crack':        'Low',
    'edge':         'Low',
}

def detect(image: Image.Image, conf: float, iou: float):
    if image is None:
        return None, " Please upload an image."

    img_np  = np.array(image.convert("RGB"))
    t0      = time.perf_counter()
    results = model.predict(img_np, conf=conf, iou=iou, verbose=False)[0]
    elapsed = time.perf_counter() - t0

    out   = image.convert("RGB").copy()
    draw  = ImageDraw.Draw(out)
    counts     = {}
    det_lines  = []

    if results.boxes is not None and len(results.boxes):
        for box in results.boxes:
            cls_id   = int(box.cls.item())
            conf_    = float(box.conf.item())
            cls_name = CLASS_NAMES[cls_id]
            color    = CLASS_COLORS[cls_name]
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())

            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            label = f"{cls_name} {conf_:.2f}"
            tw    = len(label) * 7
            draw.rectangle([x1, y1-20, x1+tw+4, y1], fill=color)
            draw.text((x1+3, y1-18), label, fill=(0, 0, 0))

            counts[cls_name] = counts.get(cls_name, 0) + 1
            det_lines.append(
                f"| {cls_name} | {conf_:.2f} | {SEVERITY[cls_name]} |"
            )

    n   = sum(counts.values())
    fps = 1.0 / elapsed if elapsed > 0 else 0

    if n == 0:
        summary = "### No road damage detected above threshold."
    else:
        class_summary = "\n".join(
            f"- **{k}**: {v} instance{'s' if v>1 else ''} — {SEVERITY[k]}"
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        )
        table = "| Class | Confidence | Severity |\n|---|---|---|\n" + "\n".join(det_lines)

        summary = f"""###  Detection Results

**Total detections:** {n}
**Inference time:** {elapsed*1000:.1f} ms &nbsp;|&nbsp; **FPS:** {fps:.1f}

---

####  Detected Damage Types
{class_summary}

---

####  All Detections
{table}
"""
    return out, summary


with gr.Blocks(
    title="Road Damage Detector",
    theme=gr.themes.Soft(),
    css="""
        .title  { text-align: center; }
        .badge  { display: inline-block; padding: 2px 8px;
                  border-radius: 4px; font-size: 12px; }
    """
) as demo:

    gr.Markdown("""
    # Road Damage Detection System
    **Model:** YOLOv11s &nbsp;|&nbsp;
    **Dataset:** Road Damage (7 classes, ~6000 images) &nbsp;|&nbsp;
    **mAP@0.5:** 0.406 &nbsp;|&nbsp;
    **FPS:** 64.1
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(
                type="pil",
                label="Upload Road Image",
                height=350
            )
            with gr.Accordion("Detection Settings", open=True):
                conf_sl = gr.Slider(
                    0.10, 0.90, value=0.25, step=0.05,
                    label="Confidence Threshold",
                    info="Lower = more detections, higher = fewer but more confident"
                )
                iou_sl = gr.Slider(
                    0.10, 0.90, value=0.45, step=0.05,
                    label="IoU Threshold (NMS)",
                    info="Higher = less overlap suppression"
                )
            detect_btn = gr.Button(
                "Detect Road Damage",
                variant="primary",
                size="lg"
            )

    
        with gr.Column(scale=1):
            img_out  = gr.Image(type="pil", label="📊 Detection Result", height=350)
            info_out = gr.Markdown()

    detect_btn.click(
        fn=detect,
        inputs=[img_in, conf_sl, iou_sl],
        outputs=[img_out, info_out]
    )

    gr.Markdown("""
    ---
    ### Class Legend & Severity
    | Class | Severity | Description |
    |---|---|---|---|
    | pothole | Purple |  High | Holes in road surface |
    | alligator |  Red |  High | Network of cracks |
    | block |  Blue |  Medium | Block pattern cracks |
    | longitudinal | Orange | Medium | Cracks along road direction |
    | transverse | Teal | Medium | Cracks across road direction |
    | crack | Green | Low | General surface cracks |
    | edge | Yellow | Low | Edge deterioration |

    ---
    ### Model Performance
    | Metric | Score |
    |---|---|
    | mAP@0.5 | 0.406 |
    | mAP@0.5:0.95 | 0.233 |
    | Precision | 0.616 |
    | Recall | 0.383 |
    | FPS (T4 GPU) | 64.1 |
    | Model Size | 19.2 MB |
    """)

demo.launch(share=True, debug=False)
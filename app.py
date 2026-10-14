import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="HelmetGuard AI | Traffic Safety",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "model" / "best.pt"

# --------------------------------------------------
# PROFESSIONAL LIGHT UI
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
    color: #182230;
}
[data-testid="stHeader"] {
    background: #ffffff;
}
.block-container {
    max-width: 1450px;
    padding: 1.8rem 2rem 3rem;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dce4ee;
}
h1, h2, h3, p, label {
    color: #182230;
}
.hero {
    background: linear-gradient(110deg, #e8f1ff, #eaf8f0);
    border: 1px solid #cbdcf4;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 22px;
}
.subtitle {
    color: #536579;
    font-size: 15px;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #dce4ee;
    border-radius: 12px;
    padding: 16px;
    min-height: 105px;
}
.metric-label {
    color: #536579;
    font-size: 14px;
}
.metric-value {
    font-size: 29px;
    font-weight: 700;
    margin-top: 6px;
}
.green-text {
    color: #16804a;
    font-weight: 600;
}
.red-text {
    color: #c62828;
    font-weight: 600;
}
.blue-text {
    color: #175cd3;
    font-weight: 600;
}
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #84adf2 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"] section {
    background: #f7faff !important;
    border: 1px solid #d6e5fb !important;
    border-radius: 9px !important;
}
[data-testid="stFileUploader"] button {
    background: #175cd3 !important;
    color: #ffffff !important;
    border: 1px solid #175cd3 !important;
    border-radius: 7px !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #344054 !important;
}
div.stButton > button[kind="primary"] {
    background: #175cd3 !important;
    color: #ffffff !important;
    border: 1px solid #175cd3 !important;
    border-radius: 8px !important;
    min-height: 44px;
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background: #1849a9 !important;
    color: #ffffff !important;
}
.stDownloadButton button {
    background: #ffffff;
    color: #175cd3;
    border: 1px solid #84adf2;
    border-radius: 8px;
}
hr {
    border-color: #dce4ee;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(str(path))


def metric_card(label, value, colour):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{colour}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div style="font-size:13px;color:#175cd3;font-weight:700;
                letter-spacing:1px;">
        TRAFFIC SAFETY ANALYTICS
    </div>
    <div style="font-size:30px;font-weight:700;color:#172b4d;
                margin-top:7px;">
        HelmetGuard AI
    </div>
    <div style="font-size:16px;color:#475467;margin-top:8px;">
        AI-assisted helmet compliance inspection for traffic safety teams.
    </div>
    <div style="font-size:14px;color:#475467;margin-top:8px;">
        Upload traffic evidence, review model detections, inspect confidence
        scores and export an annotated image with its detection register.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Place the trained checkpoint at model/best.pt.")
    st.stop()


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------
with st.sidebar:
    st.subheader("Inspection settings")
    st.caption("Adjust the minimum confidence for each class.")

    helmet_threshold = st.slider(
        "Helmet confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05
    )

    no_helmet_threshold = st.slider(
        "No-helmet confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05
    )

    st.divider()
    st.subheader("Model information")
    st.write("**Architecture:** YOLOv8s")
    st.write("**Task:** Object detection")
    st.write("**Classes:** helmet, no_helmet")
    st.write("**Weights:** `model/best.pt`")
    st.markdown(
        '<span class="green-text">Green — Helmet</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="red-text">Red — Potential no helmet</span>',
        unsafe_allow_html=True
    )
    st.caption(
        "This app runs the selected trained checkpoint. Other model variants "
        "were used during experimentation and comparison."
    )


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------
st.subheader("1. Upload traffic evidence")
st.markdown(
    '<div class="subtitle">Choose a JPG, PNG or WEBP image containing '
    'road traffic or motorcycle riders.</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Select an image to inspect",
    type=["jpg", "jpeg", "png", "webp"],
    help="Select an image from your device.",
    key="traffic_image"
)

if uploaded is None:
    st.markdown("### Inspection workflow")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**01 — Submit evidence**")
        st.write("Upload a traffic image for review.")

    with c2:
        st.markdown("**02 — Analyse detections**")
        st.write("Run the trained detector and inspect its predictions.")

    with c3:
        st.markdown("**03 — Review and export**")
        st.write("View the marked image and download the detection register.")

    st.info(
        "Green boxes indicate helmet detections. Red boxes indicate potential "
        "no-helmet detections. Every flagged case requires human review."
    )


# --------------------------------------------------
# IMAGE PREVIEW AND PREDICTION
# --------------------------------------------------
else:
    image = Image.open(uploaded).convert("RGB")
    image_key = f"{uploaded.name}_{len(uploaded.getvalue())}"

    # Clear old predictions when a new image is selected.
    if st.session_state.get("hg_image_key") != image_key:
        st.session_state.pop("hg_result", None)
        st.session_state["hg_image_key"] = image_key

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Submitted image")
        st.image(image, use_container_width=True)
        st.caption(
            f"{uploaded.name} | {image.width} × {image.height} pixels"
        )

    with right:
        st.subheader("Ready for inspection")
        st.write(
            "The selected YOLOv8s checkpoint will locate objects belonging "
            "to its trained helmet and no-helmet classes."
        )
        st.markdown(
            '<span class="green-text">Green — Helmet</span><br>'
            '<span class="red-text">Red — Potential no helmet</span>',
            unsafe_allow_html=True
        )
        run_prediction = st.button(
            "Analyse image",
            type="primary",
            use_container_width=True
        )

    if run_prediction:
        try:
            with st.spinner("Analysing image..."):
                model = load_model(str(MODEL_PATH))
                prediction = model.predict(
                    image,
                    imgsz=640,
                    conf=0.10,
                    verbose=False
                )[0]

                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                font = ImageFont.load_default()
                rows = []

                for box in prediction.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = str(
                        model.names[class_id]
                    ).lower().replace(" ", "_")

                    if class_name == "helmet":
                        if confidence < helmet_threshold:
                            continue
                        colour = (22, 128, 74)
                        display_name = "Helmet"

                    elif class_name == "no_helmet":
                        if confidence < no_helmet_threshold:
                            continue
                        colour = (198, 40, 40)
                        display_name = "Potential no helmet"

                    else:
                        continue

                    x1, y1, x2, y2 = [
                        int(v) for v in box.xyxy[0].tolist()
                    ]

                    label = f"{display_name} | {confidence:.2f}"

                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=colour,
                        width=max(3, image.width // 300)
                    )

                    text_y = max(0, y1 - 19)
                    text_box = draw.textbbox(
                        (x1, text_y), label, font=font
                    )
                    draw.rectangle(text_box, fill=colour)
                    draw.text(
                        (x1, text_y),
                        label,
                        fill="white",
                        font=font
                    )

                    rows.append({
                        "Predicted class": display_name,
                        "Confidence": round(confidence, 3),
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2,
                        "Review status": (
                            "Needs human review"
                            if class_name == "no_helmet"
                            else "Helmet detected"
                        )
                    })

            st.session_state["hg_result"] = {
                "image": annotated,
                "rows": rows,
                "filename": uploaded.name,
                "image_key": image_key
            }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# RESULTS, EXPORTS AND ANALYTICS
# --------------------------------------------------
result = st.session_state.get("hg_result")

if result and uploaded is not None:
    current_key = f"{uploaded.name}_{len(uploaded.getvalue())}"

    if result["image_key"] == current_key:
        rows = result["rows"]

        helmet_count = sum(
            row["Predicted class"] == "Helmet" for row in rows
        )
        no_helmet_count = sum(
            row["Predicted class"] == "Potential no helmet" for row in rows
        )

        st.divider()
        st.subheader("2. Inspection summary")

        a, b, c = st.columns(3)
        with a:
            metric_card("Helmet detections", helmet_count, "#16804a")
        with b:
            metric_card("Potential no-helmet cases", no_helmet_count, "#c62828")
        with c:
            metric_card("Total detections", len(rows), "#175cd3")

        st.write("")
        if no_helmet_count:
            st.error(
                f"{no_helmet_count} potential no-helmet detection(s) flagged. "
                "Review the red boxes and original evidence before action."
            )
        elif helmet_count:
            st.success(
                "Helmet detections found. No potential no-helmet cases passed "
                "the selected threshold. This does not prove full compliance."
            )
        else:
            st.warning(
                "No detections passed the selected thresholds. The image may "
                "still contain riders or violations that the model missed."
            )

        st.subheader("3. Annotated evidence")
        st.image(
            result["image"],
            caption="Green: helmet | Red: potential no helmet",
            use_container_width=True
        )

        image_buffer = io.BytesIO()
        result["image"].save(image_buffer, format="PNG")

        df = pd.DataFrame(rows)
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download annotated image",
                data=image_buffer.getvalue(),
                file_name="helmetguard_evidence.png",
                mime="image/png",
                use_container_width=True
            )
        with d2:
            st.download_button(
                "Export detection register (CSV)",
                data=csv_bytes,
                file_name="helmetguard_detection_register.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.subheader("4. Detection register")
        if rows:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No detections met the selected confidence thresholds.")

        # --------------------------------------------------
        # ANALYTICS
        # --------------------------------------------------
        st.divider()
        st.subheader("5. Traffic safety analytics")
        st.caption(
            "These charts summarise the current image's detections. "
            "They do not estimate city-wide violation rates."
        )

        if rows:
            chart1, chart2 = st.columns(2, gap="large")

            # Detection distribution
            with chart1:
                st.markdown("**Detection distribution**")

                counts = df["Predicted class"].value_counts()
                bar_colours = [
                    "#16804a" if name == "Helmet" else "#c62828"
                    for name in counts.index
                ]

                fig, ax = plt.subplots(figsize=(6, 4))
                counts.plot(
                    kind="bar",
                    ax=ax,
                    color=bar_colours
                )
                ax.set_ylabel("Number of detections")
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Confidence scores
            with chart2:
                st.markdown("**Confidence by detection**")

                plot_df = df.reset_index(drop=True).copy()
                plot_df["Detection number"] = np.arange(
                    1, len(plot_df) + 1
                )

                bar_colours = [
                    "#16804a" if name == "Helmet" else "#c62828"
                    for name in plot_df["Predicted class"]
                ]

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(
                    plot_df["Detection number"],
                    plot_df["Confidence"],
                    color=bar_colours
                )
                ax.set_xlabel("Detection number")
                ax.set_ylabel("Confidence score")
                ax.set_ylim(0, 1)
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Spatial heatmap of detection centres
            st.markdown("**Detection location heatmap**")
            st.caption(
                "Brighter cells show where more detected box centres fall "
                "within this image. This is not a geographic map."
            )

            x_centres = ((df["X1"] + df["X2"]) / 2) / image.width
            y_centres = ((df["Y1"] + df["Y2"]) / 2) / image.height

            heatmap, _, _ = np.histogram2d(
                y_centres,
                x_centres,
                bins=(10, 10),
                range=[[0, 1], [0, 1]]
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(
                image,
                extent=[0, 1, 1, 0],
                aspect="auto",
                alpha=0.50
            )
            heat = ax.imshow(
                heatmap,
                extent=[0, 1, 1, 0],
                cmap="YlOrRd",
                alpha=0.70,
                interpolation="bilinear"
            )
            ax.set_title("Spatial distribution of detected objects")
            ax.set_xlabel("Relative horizontal image position")
            ax.set_ylabel("Relative vertical image position")
            fig.colorbar(heat, ax=ax, label="Detection density")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info(
                "Analytics will appear when at least one detection passes "
                "the selected confidence thresholds."
            )

        st.caption(
            "HelmetGuard AI is a decision-support prototype. Predictions can "
            "be incorrect or missed. Red boxes are potential cases, not "
            "verified legal violations. Review the original evidence before "
            "taking action. Boxes represent detections, not unique-person tracking."
        )

st.divider()
st.caption(
    "HelmetGuard AI | Traffic Safety Decision-Support Prototype"
)

import streamlit as st
import subprocess
import json
import io
import zipfile
from PIL import Image, ImageDraw
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="📜  Old Nepali OCR - Segmentaion", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] > div:first-child { width: 350px; }
    [data-testid="stFileUploader"] > div { width: 100% !important; }
    [data-testid="stButton"] button { width: 100% !important; }
    .stSlider > div > div:first-child { height: 8px !important; border-radius: 4px; }
    .stButton > button {
        background-color: #00509D; color: #FFFFFF;
        border-radius: 6px; padding: 8px 16px;
    }
    .stButton > button:hover { background-color: #003F7D; }
    h1, h2, h3 { font-family: 'Georgia', serif; color: #2C3E50; }
    hr { border: none; height: 1px; background-color: #BDC3C7; }
    </style>
    """,
    unsafe_allow_html=True
)

# for the sidebar
st.sidebar.header("1. Input & Segmentation")
uploaded_file = st.sidebar.file_uploader(
    "Upload image (JPEG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.session_state.img_file = uploaded_file

if st.sidebar.button("Run Segmentation"):
    if not st.session_state.get("img_file"):
        st.sidebar.error("Please upload an image first.")
    else:
        temp_path = "temp_input.png"
        with open(temp_path, "wb") as f:
            f.write(st.session_state.img_file.getbuffer())

        segmentation_json = "segmentation.json"
        with st.spinner("Running segmentation..."):
            result = subprocess.run(
                ["kraken", "-i", temp_path, segmentation_json, "segment", "-bl"],
                capture_output=True, text=True
            )
        if result.returncode != 0:
            st.sidebar.error(f"Segmentation failed: {result.stderr}")
        else:
            img = Image.open(temp_path).convert("RGB")
            seg_data = json.load(open(segmentation_json, encoding="utf-8"))

            st.session_state.img_obj = img
            st.session_state.img_arr = np.array(img)
            st.session_state.seg_data = seg_data

            orig_crops = []
            for idx, line in enumerate(seg_data.get("lines", [])):
                if "boundary" not in line:
                    continue
                xs, ys = zip(*line["boundary"])
                left, upper = int(min(xs)), int(min(ys))
                right, lower = int(max(xs)), int(max(ys))
                crop_img = img.crop((left, upper, right, lower))
                orig_crops.append((f"Line_{idx+1}", crop_img))
            st.session_state.original_crops = orig_crops

            st.sidebar.success("Segmentation complete.")

# main part
st.title("Old Nepali OCR Visualizer")
st.markdown("---")

if "img_arr" in st.session_state and "seg_data" in st.session_state:
    img = st.session_state.img_obj
    lines = st.session_state.seg_data.get("lines", [])
    orig_w, orig_h = img.size

    st.subheader("Detected Lines")
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for line in lines:
        if "boundary" not in line:
            continue
        poly = [(int(x), int(y)) for x, y in line["boundary"]]
        draw.polygon(poly, fill=(255, 255, 0, 80))
    composite = Image.alpha_composite(base, overlay).convert("RGB")
    st.session_state.adjusted_overlay = composite

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(composite, caption="Overlay of detected text lines", width=orig_w // 2)

    st.subheader("2. Adjust Segmentation")

    with st.expander("Please remove Unwanted Lines", expanded=True):
        chart_height = st.slider("Panel Height (px)", 200, 1200, 600)
        padding = st.slider("Add Padding to Selected Lines (pixels)", 0, 100, 10)

        labels = [f"Line {i+1}" for i in range(len(lines))]
        to_remove = st.multiselect("Remove Lines:", labels, default=[])
        padding_lines_labels = st.multiselect("Apply padding to Lines:", labels, default=[])

        st.markdown("To find the line numbers, hover over the lines in the plot below.")

        remove_idx = {int(lbl.split()[1]) - 1 for lbl in to_remove}
        padding_idx = {int(lbl.split()[1]) - 1 for lbl in padding_lines_labels}

        with st.spinner("Please wait, generating preview..."):
            fig = px.imshow(st.session_state.img_arr, origin="upper")
            fig.update_layout(height=chart_height, margin=dict(l=0, r=0, t=30, b=0))
            fig.update_coloraxes(showscale=False)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False, autorange="reversed")

            palette = px.colors.qualitative.Plotly

            for idx, line in enumerate(lines):
                if idx in remove_idx or "boundary" not in line:
                    continue
                xs, ys = zip(*line["boundary"])
                color = palette[idx % len(palette)]
                fig.add_scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=2),
                    customdata=[idx+1]*len(xs),
                    hovertemplate="Line %{customdata}<extra></extra>",
                    showlegend=False
                )
                l, r, t, b = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
                if idx in padding_idx:
                    pad_l = max(0, l - padding)
                    pad_r = min(orig_w, r + padding)
                    pad_t = max(0, t - padding)
                    pad_b = min(orig_h, b + padding)
                    fig.add_shape(
                        type="rect", x0=pad_l, y0=pad_b, x1=pad_r, y1=pad_t,
                        fillcolor=color, opacity=0.35, line=dict(width=2, color='black')
                    )
                else:
                    fig.add_shape(
                        type="rect", x0=l, y0=b, x1=r, y1=t,
                        fillcolor=color, opacity=0.25, line=dict(width=1, color='gray')
                    )
            st.plotly_chart(fig, use_container_width=True)

    
    orig_crops, adj_crops = [], []
    for idx, line in enumerate(lines):
        if idx in remove_idx or "boundary" not in line:
            continue
        xs, ys = zip(*line["boundary"])
        if idx in padding_idx:
            left  = max(0, int(min(xs)) - padding)
            upper = max(0, int(min(ys)) - padding)
            right = min(orig_w, int(max(xs)) + padding)
            lower = min(orig_h, int(max(ys)) + padding)
        else:
            left  = int(min(xs))
            upper = int(min(ys))
            right = int(max(xs))
            lower = int(max(ys))
        crop_img = img.crop((left, upper, right, lower))
        orig_crops.append((f"Line_{idx+1}", crop_img))
        adj_crops.append((f"Line_{idx+1}", crop_img))

    st.session_state.original_crops = orig_crops
    st.session_state.adjusted_crops = adj_crops
    st.session_state.adjusted_overlay = composite

    if "adjusted_saved" not in st.session_state:
        st.session_state.adjusted_saved = False

    st.subheader("Do you want to save these changes?")
    if st.button("Save Segmentations"):
        new_lines = []
        for idx, line in enumerate(lines):
            if idx in remove_idx or "boundary" not in line:
                continue
            xs, ys = zip(*line["boundary"])
            line["boundary"] = [[int(x), int(y)] for x, y in zip(xs, ys)]
            new_lines.append(line)
        new_seg_data = st.session_state.seg_data.copy()
        new_seg_data["lines"] = new_lines
        with open("segmentation.json", "w", encoding="utf-8") as f:
            json.dump(new_seg_data, f, ensure_ascii=False, indent=2)
        st.session_state.seg_data = new_seg_data
        st.session_state.adjusted_saved = True
        st.success("Saved adjusted segmentations successfully!")

    if st.session_state.adjusted_saved:
        if st.button("Proceed to Prediction ➡️"):
            st.switch_page("pages/2_prediction.py")
    else:
        st.info("Please save your changes before proceeding to prediction.")
        

    # st.markdown("---")
    # st.subheader("3. Choose Segmentation for Prediction")
    # choice = st.selectbox("Pick which crops to use for OCR prediction:", ["Original Segmentation", "Adjusted Segmentation"])

    # st.session_state.prediction_crops = st.session_state.original_crops if choice == "Original Segmentation" else st.session_state.adjusted_crops
    # st.session_state.prediction_overlay = st.session_state.adjusted_overlay
    st.session_state.crops = st.session_state.adjusted_crops 
    st.session_state.segmentation_overlay = composite

    # download part
    st.markdown("---")
    st.subheader("Download Segmentation Data")

    buf_orig = io.BytesIO()
    with zipfile.ZipFile(buf_orig, "w") as zf:
        zf.writestr("segmentation.json", json.dumps(st.session_state.seg_data, ensure_ascii=False, indent=2))
        for name, img_crop in st.session_state.original_crops:
            img_bytes = io.BytesIO()
            img_crop.save(img_bytes, format="PNG")
            zf.writestr(f"crops/{name}.png", img_bytes.getvalue())
    buf_orig.seek(0)
    st.download_button("Download Original Segmentation (.json & Crops)", data=buf_orig.getvalue(), file_name="original_segmentation_and_crops.zip", mime="application/zip")

    buf_adj = io.BytesIO()
    with zipfile.ZipFile(buf_adj, "w") as zf:
        zf.writestr("segmentation.json", json.dumps(st.session_state.seg_data, ensure_ascii=False, indent=2))
        config = {
            "panel_height": chart_height,
            "removed_lines": list(remove_idx),
            "padding": padding,
            "padding_applied_to": sorted(list(padding_idx))
        }
        zf.writestr("adjusted_settings.json", json.dumps(config, ensure_ascii=False, indent=2))
        overlay_bytes = io.BytesIO()
        st.session_state.adjusted_overlay.save(overlay_bytes, format="PNG")
        zf.writestr("adjusted_overlay.png", overlay_bytes.getvalue())
        for name, img_crop in st.session_state.adjusted_crops:
            img_bytes = io.BytesIO()
            img_crop.save(img_bytes, format="PNG")
            zf.writestr(f"crops/{name}.png", img_bytes.getvalue())
    buf_adj.seek(0)
    st.download_button("Download Adjusted Segmentation (.json & Crops)", data=buf_adj.getvalue(), file_name="adjusted_segmentation_and_crops.zip", mime="application/zip")

else:
    st.info("Please upload an image and run segmentation to begin")

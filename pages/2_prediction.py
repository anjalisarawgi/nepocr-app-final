# pages/2_prediction.py
import streamlit as st
from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast, TrOCRProcessor
import torch
from PIL import Image
import io, csv

@st.cache_resource
def load_model():
    model_path = "model-name-anonymized" # changed for now
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, processor.feature_extractor, device

def predict_from_image(image, model, tokenizer, feature_extractor, device):
    image = image.convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output = model.generate(
            pixel_values,
            max_length=256,
            num_beams=6,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            diversity_penalty=0.5,
            num_beam_groups = 2
        )
    # output is a batch of size 1
    seq = output[0]
    text = tokenizer.decode(seq, skip_special_tokens=True)
    return text

def main():
    st.header("📜 Old Nepali OCR – Prediction")

    overlay = st.session_state.get("segmentation_overlay")
    crops   = st.session_state.get("crops", [])

    col_left, col_right = st.columns(2)

    # LEFT: segmentation overlay
    with col_left:
        st.subheader("Segmentation Overlay")
        if overlay:
            st.image(overlay, use_container_width=True)
        else:
            st.info("No overlay found. Run Segmentation. Please Crop first.")

    # RIGHT: OCR controls & output
    with col_right:
        st.subheader("OCR Predictions")

        if not crops:
            st.info("Please run Segmentation first.")
            return

        choices   = ["All"] + [name for name, _ in crops]
        selection = st.selectbox("Which line(s) to OCR?", choices)

        model, tokenizer, feat_ext, device = load_model()

        if st.button("Run OCR & Show"):
            to_run = (
                crops
                if selection == "All"
                else [crops[choices.index(selection) - 1]]
            )


            results = []
            for fname, img in to_run:
                txt = predict_from_image(img, model, tokenizer, feat_ext, device)
                results.append((fname, txt))
                st.markdown(
                    f"""
                    <div style="
                      border:1px solid #ccc;
                      border-radius:6px;
                      padding:12px;
                      margin-bottom:10px;
                      background-color:#fafafa;
                    ">
                      {txt}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            txt_buf = io.StringIO()
            for fname, txt in results:
                txt_buf.write(f"{txt}\n")

            txt_data = txt_buf.getvalue().encode("utf-8")

            st.download_button(
                label="Download predictions",
                data=txt_data,
                file_name="ocr_results.txt",
                mime="text/plain",
            )

if __name__ == "__main__":
    main()

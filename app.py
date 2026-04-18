# app.py
import streamlit as st

st.set_page_config(
    page_title="Old Nepali OCR",
    layout="wide"
)

st.title("📜  Old Nepali OCR")

st.markdown(
    """
    Hi, 

    **Step 1:** Switch to the segmentation tab to get started

    **Step 2:** Upload your image on the left side bar and click on "Run Segmentation" to process the image.

    **Step 3:** Once the segmentation is done, you may also choose to adjust the segmentations. You have two options for now, removing unwanted segments, and adjusting the bounding boxes of the segments with padding. 
    
    **Step 4:** Once you are satisfied with the segmentations, save the segmentations by clicking "Save Segmentations" button.
    
    **Step 5:** Switch to the prediction tab to run OCR on the segments.
    
    *Please note: Analysis is yet to be added, and more fixes can be done as required. *
    """
)

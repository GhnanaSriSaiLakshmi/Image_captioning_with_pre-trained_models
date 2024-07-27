import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set the page configuration
st.set_page_config(
    page_title="Image Captioning App",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Image Captioning App 📸")
st.write(
    """
    Upload an image and get 2-3 captions describing the content of the image.
    """
)

# Sidebar for image upload and options
st.sidebar.header("Image Upload")
with st.sidebar.expander("Upload Image"):
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.sidebar.header("Settings")
with st.sidebar.expander("Caption Settings"):
    num_captions = st.slider("Number of Captions", 1, 5, 3)
    #max_length = st.slider("Max Caption Length", 10, 50, 16)

# Initialize the image captioning model
@st.cache_resource(show_spinner=False)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

def generate_captions(image, num_captions):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, num_beams=5, num_return_sequences=num_captions)
    captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

# Main content
if uploaded_image is not None:
    # Display uploaded image with reduced size
    image = Image.open(uploaded_image)
    max_image_size = (500, 500)  # Maximum dimensions for the displayed image
    image.thumbnail(max_image_size)  # Reduce image size while maintaining aspect ratio

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate captions
    with st.spinner("Generating captions..."):
        captions = generate_captions(image, num_captions)

    st.write("## Captions:")
    for i, caption in enumerate(captions, 1):
        # Capitalize the first word of the caption
        capitalized_caption = caption.capitalize()
        st.markdown(f"*Caption {i}:* {capitalized_caption}")

else:
    st.info("Please upload an image to generate captions.")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #333;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    .caption {
        font-size: 16px;
        color: #4CAF50;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    """
    <div class="footer">
        <p></p>
    </div>
    """,
    unsafe_allow_html=True,
)
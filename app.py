### ADD YOUR CODE HERE ###
import streamlit as st
from transformers import AutoModelForImageClassification 
import torch
from PIL import Image
import torchvision.transforms as transforms


@st.cache_resource
def load_model(model_path):
    try:
        model = AutoModelForImageClassification.from_pretrained(model_path)
        print(f"Successfully loaded model from: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

model = AutoModelForImageClassification.from_pretrained("./model")

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

id2label = model.config.id2label if model and hasattr(model, 'config') and hasattr(model.config, 'id2label') else None
st.title("ViT Cat vs. Dog Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None and id2label is not None:
    image = Image.open(uploaded_file)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")

    if st.button('Classify Image'):
        st.write("Classifying...")
        try:
            input_tensor = inference_transform(image)
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                input_batch = input_batch.to('cpu')
                model.to('cpu')

                outputs = model(pixel_values=input_batch).logits
                logits = outputs

            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = id2label.get(predicted_class_idx, f"Unknown Label ({predicted_class_idx})")
            probabilities = torch.softmax(logits, dim=-1)
            confidence = probabilities[0, predicted_class_idx].item() * 100

            label_mapping = {"LABEL_0": "Cat", "LABEL_1": "Dog"}

            st.success(f"Prediction: **{label_mapping[predicted_label]}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

        except Exception as e:
            st.error(f"An error occurred during processing/prediction: {e}")
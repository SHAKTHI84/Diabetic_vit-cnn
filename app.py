import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ViTForImageClassification

# Set page configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Detector",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CNN Model Definition
class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 25 * 25, 32)
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # First block
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        
        # Second block
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        
        # Third block
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)
        
        # Flatten and fully connected
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Hybrid Ensemble Model
class HybridEnsemble:
    def __init__(self, cnn_model, vit_model, device):
        self.cnn_model = cnn_model
        self.vit_model = vit_model
        self.device = device
        
        # Set models to evaluation mode
        self.cnn_model.eval()
        self.vit_model.eval()
    
    def predict(self, images):
        """Return class predictions"""
        probs = self.predict_proba(images)
        _, predicted = torch.max(probs, 1)
        return predicted
    
    def predict_proba(self, images):
        """Return probability predictions"""
        with torch.no_grad():
            # CNN predictions
            cnn_outputs = self.cnn_model(images)
            cnn_probs = torch.softmax(cnn_outputs, dim=1)
            
            # ViT predictions
            vit_outputs = self.vit_model(images).logits
            vit_probs = torch.softmax(vit_outputs, dim=1)
            
            # Average the probabilities (simple ensemble)
            ensemble_probs = (cnn_probs + vit_probs) / 2
            
        return ensemble_probs

# Function to load models
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CNN model
    cnn_model = CustomCNN(num_classes=5)
    try:
        checkpoint = torch.load('diabetic_retinopathy_cnn.pth', map_location=device)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
    except:
        st.error("❌ Could not load CNN model. Please ensure 'diabetic_retinopathy_cnn.pth' is available.")
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    
    # Load ViT model
    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    try:
        vit_model.load_state_dict(torch.load('diabetic_retinopathy_vit.pth', map_location=device))
    except:
        st.error("❌ Could not load ViT model. Please ensure 'diabetic_retinopathy_vit.pth' is available.")
    vit_model = vit_model.to(device)
    vit_model.eval()
    
    # Create ensemble
    ensemble = HybridEnsemble(cnn_model, vit_model, device)
    
    return {
        'CNN': cnn_model,
        'ViT': vit_model,
        'Ensemble': ensemble
    }, device

# Function to preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to make prediction
def predict_image(image, model, model_type, device):
    # Preprocess the image
    image_tensor = preprocess_image(image).to(device)
    
    # Get prediction
    with torch.no_grad():
        if model_type == 'CNN':
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
        elif model_type == 'ViT':
            outputs = model(image_tensor).logits
            probs = torch.softmax(outputs, dim=1)
        else:  # Ensemble
            probs = model.predict_proba(image_tensor)
    
    return probs.cpu().numpy()[0]

# Class names
class_names = [
    "Mild DR (0)",      # Index 0
    "Moderate DR (1)",  # Index 1
    "No DR (2)",        # Index 2
    "Severe DR (3)",    # Index 3
    "Proliferative DR (4)"  # Index 4
]

# Class descriptions
class_descriptions = {
    "No DR (2)": "No signs of diabetic retinopathy detected in the retina.",
    "Mild DR (0)": "Presence of microaneurysms, the earliest clinical sign of diabetic retinopathy.",
    "Moderate DR (1)": "More microaneurysms, dot and blot hemorrhages, venous beading, and cotton wool spots.",
    "Severe DR (3)": "Severe intraretinal hemorrhages, definite venous beading, and intraretinal microvascular abnormalities.",
    "Proliferative DR (4)": "Growth of new blood vessels on the retina and posterior surface of the vitreous, which can lead to blindness if untreated."
}

# Severity colors
severity_colors = {
    "No DR (2)": "#4CAF50",  # Green
    "Mild DR (0)": "#8BC34A",  # Light Green
    "Moderate DR (1)": "#FFC107",  # Amber
    "Severe DR (3)": "#FF9800",  # Orange
    "Proliferative DR (4)": "#F44336"  # Red
}

# Create visualization of prediction
def visualize_prediction(probabilities):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create horizontal bar chart
    colors = [severity_colors[class_name] for class_name in class_names]
    bars = ax.barh(class_names, probabilities * 100, color=colors)
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                va='center')
    
    ax.set_xlabel('Probability (%)')
    ax.set_title('Prediction Probabilities', fontsize=15)
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    return fig

# Main app
def main():
    # Sidebar
    st.sidebar.image("https://img.freepik.com/free-vector/eye-scan-concept-illustration_114360-1427.jpg", width=250)
    st.sidebar.title("🔍 Diabetic Retinopathy AI")
    
    # Fixed to ensemble model
    model_key = "Ensemble"
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 About Diabetic Retinopathy")
    st.sidebar.markdown("""
    Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes.
    It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
    
    DR is classified into stages based on severity:
    - **No DR**: No abnormalities
    - **Mild DR**: Microaneurysms only
    - **Moderate DR**: More microaneurysms, hemorrhages, hard exudates
    - **Severe DR**: Intraretinal hemorrhages, venous beading
    - **Proliferative DR**: Neovascularization, vitreous hemorrhage
    """)
    
    # Main content
    st.title("👁️ Diabetic Retinopathy Detection")
    st.markdown("""
    ### Upload a retinal image for automatic classification
    This application uses advanced deep learning models to detect signs of diabetic retinopathy from retinal images.
    """)
    
    # Load models
    with st.spinner("Loading models... This may take a moment."):
        models, device = load_models()
    
    # Create two columns for upload and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Upload Image")
        uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
            
            # Make prediction button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing with AI ensemble model..."):
                    # Make prediction
                    probabilities = predict_image(image, models[model_key], model_key, device)
                    predicted_class = np.argmax(probabilities)
                    
                    # Store results in session state for the other column to access
                    st.session_state.probabilities = probabilities
                    st.session_state.predicted_class = predicted_class
                    st.session_state.has_prediction = True
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if 'has_prediction' in st.session_state and st.session_state.has_prediction:
            # Display prediction
            probabilities = st.session_state.probabilities
            predicted_class = st.session_state.predicted_class
            
            # Result box with appropriate color
            result_color = severity_colors[class_names[predicted_class]]
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {result_color}30; border: 2px solid {result_color}">
                <h3 style="color: {result_color}">Diagnosis: {class_names[predicted_class]}</h3>
                <p>{class_descriptions[class_names[predicted_class]]}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display visualization
            st.pyplot(visualize_prediction(probabilities))
            
            # Show recommendation based on severity
            st.subheader("📝 Recommendation")
            if predicted_class == 2:
                st.success("✅ No signs of diabetic retinopathy detected. Continue routine eye screenings.")
            elif predicted_class == 0:
                st.info("ℹ️ Mild DR detected. Follow up with your ophthalmologist within 6-12 months.")
            elif predicted_class == 1:
                st.warning("⚠️ Moderate DR detected. Follow up with your ophthalmologist within 3-6 months.")
            elif predicted_class == 3:
                st.warning("⚠️ Severe DR detected. Follow up with your ophthalmologist within 1-3 months.")
            else:
                st.error("🚨 Proliferative DR detected. Urgent referral to an ophthalmologist is recommended.")
                
            # Disclaimer
            st.markdown("---")
            st.caption("⚠️ **Disclaimer**: This is an AI prediction tool and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.")
        else:
            # Placeholder before prediction
            st.info("👈 Upload an image and click 'Analyze Image' to see results here")
            
            # Sample images showing each class
            

    # Additional information section
    st.markdown("---")
    st.subheader("ℹ️ About the AI Model")
    
    # Model info
    st.markdown("""
    #### 🧠 Advanced Ensemble Model

    This application uses a hybrid ensemble model that combines two powerful deep learning architectures:

    **Convolutional Neural Network (CNN)**
    - Specialized in detecting local patterns and features
    - Processes images through multiple layers of filters
    - Effective at identifying specific retinal abnormalities

    **Vision Transformer (ViT)**
    - Analyzes relationships between different parts of the image
    - Excels at capturing global context and patterns
    - Based on Google's ViT architecture

    By combining these approaches, our ensemble model achieves:
    - Higher accuracy than either model alone
    - More reliable predictions across different image qualities
    - Better detection of subtle signs of diabetic retinopathy
    """)

if __name__ == "__main__":
    main()
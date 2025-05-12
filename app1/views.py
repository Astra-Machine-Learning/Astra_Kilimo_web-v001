import os
import base64
import json
import markdown
from PIL import Image

from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from decouple import config
import google.generativeai as genai

import torch
from torchvision import transforms

from torchvision.models import resnet18

from torch import nn

# === Initialize Gemini API ===
genai.configure(api_key=config("GOOGLE_GENAI_API_KEY"))

# === Load PyTorch Model ===

# Define the model architecture
class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Assuming 3 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instantiate and load the model
model = PlantDiseaseModel()
try:
    model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu')), strict=False)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Model loading failed")

# === Helper: Run Local Model ===
def run_local_model(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure size matches model input size
        transforms.ToTensor(),
    ])
    try:
        image = Image.open(image_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        classes = ["Healthy", "Early Blight", "Late Blight"]  # Example classes
        return classes[predicted.item()]
    except Exception as e:
        raise RuntimeError(f"Error running local model: {e}")

# === Helper: Use Gemini to Explain Diagnosis ===
def send_to_gemini(diagnosis):
    prompt = f"A local model detected the plant issue as '{diagnosis}'. Can you explain this in simple terms and suggest remedies?"
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content([{"text": prompt}])
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error interacting with Gemini: {e}")

# === Optional: Image to Base64 ===
def encode_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error encoding image: {e}")

# === Use Gemini for Full Image Analysis ===
def analyze_image(image_path, prompt="Analyze this plant image..."):
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        encoded_image = encode_image(image_path)
        response = gemini_model.generate_content([
            {"mime_type": "image/png", "data": encoded_image},
            {"text": prompt}
        ])
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error analyzing image with Gemini: {e}")

# === Use Gemini to Analyze Symptoms ===
def analyze_symptoms(symptoms):
    prompt = f"Analyze the following symptoms and provide a diagnosis or recommendation: {', '.join(symptoms)}."
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content([{"text": prompt}])
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error analyzing symptoms with Gemini: {e}")

# === Views ===

def index(request):
    return render(request, 'app1/index.html')

def home(request):
    return render(request, 'app1/home.html')

def symptom_checker(request):
    if request.method == "POST":
        selected_symptoms = request.POST.getlist("symptoms")

        if not selected_symptoms:
            messages.error(request, "Please select at least one symptom.")
            return render(request, "app1/symptom_checker.html")

        try:
            analysis_result = analyze_symptoms(selected_symptoms)
            return render(request, "app1/symptom_checker.html", {"analysis_result": analysis_result})
        except Exception as e:
            messages.error(request, f"Error analyzing symptoms: {e}")
            return redirect("symptom_checker")

    return render(request, "app1/symptom_checker.html")

def crop_analysis(request):
    if request.method == "POST" and request.FILES.get("xray"):
        xray = request.FILES["xray"]
        fs = FileSystemStorage()
        file_path = fs.save(xray.name, xray)
        file_path = fs.path(file_path)

        try:
            # Step 1: Local model
            diagnosis = run_local_model(file_path)

            # Step 2: Gemini explanation
            explanation = send_to_gemini(diagnosis)

            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

            return render(request, "app1/crop_analysis.html", {
                "diagnosis": diagnosis,
                "analysis_result": markdown.markdown(explanation)
            })
        except Exception as e:
            messages.error(request, f"Error analyzing image: {e}")
            return redirect("crop_analysis")

    return render(request, "app1/crop_analysis.html")

def community(request):
    return render(request, 'app1/community.html')

def chat(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input")
        if not user_input:
            messages.error(request, "Please enter a message.")
            return render(request, "app1/chat.html")

        try:
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            response = gemini_model.generate_content([{"text": user_input}])
            html_response = markdown.markdown(response.text)
            return render(request, "app1/chat.html", {"response": html_response})
        except Exception as e:
            messages.error(request, f"Error chatting with AI: {e}")
            return redirect("chat")

    return render(request, "app1/chat.html")

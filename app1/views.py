import os
import base64
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from decouple import config
import google.generativeai as genai
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import markdown

# Configure Google Gemini API using the API key from the .env file
genai.configure(api_key=config("GOOGLE_GENAI_API_KEY"))

def encode_image(image_path):
    """Encodes an image file to base64 format."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def analyze_image(image_path, prompt="Analyze this plant image to identify any diseases, pests, or nutrient deficiencies. Provide actionable advice in simple language for farmers."):
    """Sends an image along with a prompt to Google Gemini API for analysis."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    encoded_image = encode_image(image_path)
    response = model.generate_content([
        {"mime_type": "image/png", "data": encoded_image},
        {"text": prompt}
    ])
    return response.text

def analyze_symptoms(symptoms):
    """Sends a text-based prompt to Google Gemini API for symptom analysis."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Analyze the following symptoms and provide a diagnosis or recommendation: {', '.join(symptoms)}."
    response = model.generate_content([{"text": prompt}])
    return response.text

def index(request):
    return render(request, 'app1/index.html')

def home(request):
    return render(request, 'app1/home.html')

def symptom_checker(request):
    if request.method == "POST":
        # Get selected symptoms from the form
        selected_symptoms = request.POST.getlist("symptoms")
        
        if not selected_symptoms:
            messages.error(request, "Please select at least one symptom.")
            return render(request, "app1/symptom_checker.html")

        try:
            # Use the Gemini API to analyze the symptoms
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
            result = analyze_image(file_path)
            os.remove(file_path)  # Clean up the uploaded file
            # Convert markdown to HTML
            html_result = markdown.markdown(result)
            return render(request, "app1/crop_analysis.html", {"analysis_result": html_result})
        except Exception as e:
            messages.error(request, f"Error analyzing image: {e}")
            return redirect("crop_analysis")

    return render(request, "app1/crop_analysis.html")

def community(request):
    return render(request, 'app1/community.html')


# chat with ai - 
def chat(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input")
        if not user_input:
            messages.error(request, "Please enter a message.")
            return render(request, "app1/chat.html")

        try:
            # Use the Gemini API to get a response
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([{"text": user_input}])
            return render(request, "app1/chat.html", {"response": response.text})
        except Exception as e:
            messages.error(request, f"Error chatting with AI: {e}")
            return redirect("chat")

    return render(request, "app1/chat.html")
# Astra Kilimo 🌱

Astra Kilimo is a Django-based web application developed to help farmers and agricultural researchers detect plant diseases using machine learning. The project integrates Google Gemini API to provide intelligent suggestions and insights. The plant disease detection model is trained using publicly available datasets from Kaggle.

---

## 🚀 Getting Started

### 📦 Clone the Repository

```bash
git clone https://github.com/Astra-Machine-Learning/Astra_Kilimo_web-v001.git
cd Astra_Kilimo_web-v001
```

### 🐍 Set Up Virtual Environment (optional but recommended)

```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

### 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔐 Google Gemini API Key

Astra Kilimo uses the Google Gemini API to provide contextual insights and assistance:

1. Go to [Google AI Studio](https://makersuite.google.com/app) and sign in.
2. Create an API key.
3. Add your API key to your environment:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Alternatively, create a `.env` file in the project root:

```
GEMINI_API_KEY=your-api-key-here
```

### 🔄 Apply Migrations & Run the Server

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Access the application in your browser at `http://127.0.0.1:8000`

---

## 📊 Dataset Used

We use the **PlantVillage Dataset** from Kaggle, which contains 50,000+ images of diseased and healthy plant leaves.

* Source: [Plant Disease Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

To use the dataset:

1. Download the dataset from the link above.
2. Unzip and place the images in a `media/plant_images/` directory or as specified in the Django project settings.

---

## 🤖 Features

* Upload leaf images for disease prediction
* AI-generated recommendations powered by Google Gemini
* Django admin for dataset management
* Clean, responsive web interface

---

## 🛠 Technologies Used

* Python & Django
* Google Gemini API
* TensorFlow / PyTorch (model integration planned)
* HTML/CSS/Bootstrap

---

## 🤝 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📧 Contact

For more information, reach out to us at: [info.astrasoft@gmail.com](mailto:info.astrasoft@gmail.com)

> Built with 💚 by Astra Softwares – *Transforming the Digital World*


from django.contrib import admin
from django.urls import path
from app1.views import index, symptom_checker, crop_analysis, chat, home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('home/', home, name='home'),  # Corrected
    path('symptom_checker/', symptom_checker, name='symptom_checker'),  # Corrected
    path('crop_analysis/', crop_analysis, name='crop_analysis'),  # Corrected
    path('chat/', chat, name='chat'),  # Corrected
]

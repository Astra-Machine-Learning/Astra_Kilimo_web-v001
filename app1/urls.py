from django.contrib import admin
from django.urls import path
from app1 import views

# add chat.html path 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.crop_analysis, name='crop_analysis'),
    path('chat/', views.chat, name='chat'),
]
# app/face_recognition/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/face/recognize/$', consumers.FaceRecognitionConsumer.as_asgi()),
]
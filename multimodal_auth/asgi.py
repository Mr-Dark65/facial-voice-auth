"""
ASGI config for multimodal_auth project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""
# multimodal_auth/asgi.py
import os
import django

# Configurar Django ANTES de importar cualquier cosa
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'multimodal_auth.settings')
django.setup()  # ← AGREGAR ESTO

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from app.face_recognition.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})

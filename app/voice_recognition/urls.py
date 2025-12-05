from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import VoiceSampleViewSet, VoiceRecognitionViewSet, VoiceRecognitionModelViewSet

router = DefaultRouter()
router.register(r'samples', VoiceSampleViewSet, basename='voice-sample')
router.register(r'models', VoiceRecognitionModelViewSet, basename='voice-model')
router.register(r'', VoiceRecognitionViewSet, basename='voice-recognition')

urlpatterns = [
    path('', include(router.urls)),
]

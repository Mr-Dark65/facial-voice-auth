from django.urls import path
from .views import (
    UploadFaceImageView, RegisterFaceView, FaceImagesListView,
    DeleteFaceImageView, TrainFaceModelView, DatasetStatsView,
    FaceRecognitionView, FaceVerificationView, ModelListView
)

app_name = 'face_recognition'

urlpatterns = [
    # Registro de imágenes faciales
    path('upload/', UploadFaceImageView.as_view(), name='upload-image'),
    path('register/', RegisterFaceView.as_view(), name='register-face'),
    path('my-images/', FaceImagesListView.as_view(), name='my-images'),
    path('images/<int:pk>/', DeleteFaceImageView.as_view(), name='delete-image'),
    
    # Reconocimiento y verificación
    path('recognize/', FaceRecognitionView.as_view(), name='recognize'),
    path('verify/', FaceVerificationView.as_view(), name='verify'),
    
    # Entrenamiento y gestión de modelos
    path('train/', TrainFaceModelView.as_view(), name='train-model'),
    path('dataset-stats/', DatasetStatsView.as_view(), name='dataset-stats'),
    path('models/', ModelListView.as_view(), name='models-list'),
]

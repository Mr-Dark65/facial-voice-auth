from rest_framework import serializers
from .models import FaceImage, FaceRecognitionModel
from app.users.models import User


class FaceImageSerializer(serializers.ModelSerializer):
    """
    Serializer para imágenes faciales.
    """
    user_email = serializers.EmailField(source='user.email', read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = FaceImage
        fields = [
            'id', 'user', 'user_email', 'user_username', 'image',
            'is_valid', 'quality_score', 'face_detected',
            'uploaded_at', 'processed_at'
        ]
        read_only_fields = [
            'id', 'is_valid', 'quality_score', 'face_detected',
            'uploaded_at', 'processed_at', 'embedding'
        ]


class FaceImageUploadSerializer(serializers.Serializer):
    """
    Serializer para subir una imagen facial.
    """
    image = serializers.ImageField(required=True)
    
    def validate_image(self, value):
    
        # Límite de 5MB
        if value.size > 5 * 1024 * 1024:
            raise serializers.ValidationError("La imagen no debe superar 5MB")
        
        # Validar formato
        valid_formats = ['image/jpeg', 'image/jpg', 'image/png']
        if value.content_type not in valid_formats:
            raise serializers.ValidationError(
                "Formato no válido. Use JPEG o PNG"
            )
        
        return value


class FaceRegistrationSerializer(serializers.Serializer):
    """
    Serializer para el proceso completo de registro facial.
    """
    images = serializers.ListField(
        child=serializers.ImageField(),
        min_length=5,
        max_length=10,
        help_text="Entre 5 y 10 imágenes faciales"
    )


class FaceVerificationSerializer(serializers.Serializer):
    """
    Serializer para verificación facial.
    """
    image = serializers.ImageField(required=True)
    user_id = serializers.IntegerField(required=False)
    
    def validate_image(self, value):
        """Valida la imagen."""
        if value.size > 5 * 1024 * 1024:
            raise serializers.ValidationError("La imagen no debe superar 5MB")
        return value


class FaceRecognitionSerializer(serializers.Serializer):
    """
    Serializer para reconocimiento facial (identificación).
    """
    image = serializers.ImageField(required=True)


class FaceRecognitionModelSerializer(serializers.ModelSerializer):
    """
    Serializer para modelos de reconocimiento.
    """
    class Meta:
        model = FaceRecognitionModel
        fields = [
            'id', 'name', 'version', 'model_file',
            'accuracy', 'loss', 'val_accuracy', 'val_loss',
            'training_samples', 'epochs_trained', 'architecture',
            'is_active', 'created_at', 'trained_at'
        ]
        read_only_fields = fields


class TrainModelSerializer(serializers.Serializer):
    """
    Serializer para entrenar un modelo.
    """
    epochs = serializers.IntegerField(
        default=50,
        min_value=1,
        max_value=200,
        help_text="Número de épocas de entrenamiento"
    )
    batch_size = serializers.IntegerField(
        default=32,
        min_value=8,
        max_value=128,
        help_text="Tamaño del batch"
    )
    version = serializers.CharField(
        default='1.0',
        max_length=20,
        help_text="Versión del modelo"
    )


class DatasetStatsSerializer(serializers.Serializer):
    """
    Serializer para estadísticas del dataset.
    """
    total_users = serializers.IntegerField()
    total_images = serializers.IntegerField()
    avg_images_per_user = serializers.FloatField()
    min_images_per_user = serializers.IntegerField()
    max_images_per_user = serializers.IntegerField()
    user_ids = serializers.ListField(child=serializers.IntegerField())
    is_ready_for_training = serializers.BooleanField()


class FaceEmbeddingSerializer(serializers.Serializer):
    """
    Serializer para embeddings faciales.
    """
    embedding = serializers.ListField(
        child=serializers.FloatField(),
        read_only=True
    )
    dimension = serializers.IntegerField(read_only=True)

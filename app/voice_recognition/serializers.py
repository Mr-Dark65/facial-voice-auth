from rest_framework import serializers
from .models import VoiceSample, VoiceRecognitionModel
from app.users.models import User


class VoiceSampleSerializer(serializers.ModelSerializer):
    """Serializer para muestras de voz."""
    
    username = serializers.CharField(source='user.username', read_only=True)
    
    class Meta:
        model = VoiceSample
        fields = [
            'id', 'user', 'username', 'audio_file', 'embedding',
            'duration', 'sample_rate', 'snr', 'quality_score',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'embedding']


class VoiceUploadSerializer(serializers.Serializer):
    """Serializer para subir archivos de audio."""
    
    audio = serializers.FileField(required=True)
    user_id = serializers.IntegerField(required=False)
    
    def validate_audio(self, value):
        """Valida que el archivo sea de audio."""
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        filename = value.name.lower()
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise serializers.ValidationError(
                f"Formato no soportado. Permitidos: {', '.join(allowed_extensions)}"
            )
        
        # Validar tamaño (máximo 10MB)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("El archivo no debe exceder 10MB")
        
        return value


class VoiceRegistrationSerializer(serializers.Serializer):
    """Serializer para registro de voz de usuario."""
    
    audio = serializers.FileField(required=True)
    user_id = serializers.IntegerField(required=True)
    
    def validate_user_id(self, value):
        """Valida que el usuario exista."""
        if not User.objects.filter(id=value).exists():
            raise serializers.ValidationError("Usuario no encontrado")
        return value
    
    def validate_audio(self, value):
        """Valida el archivo de audio."""
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        filename = value.name.lower()
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise serializers.ValidationError(
                f"Formato no soportado. Permitidos: {', '.join(allowed_extensions)}"
            )
        
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("El archivo no debe exceder 10MB")
        
        return value


class VoiceRecognitionSerializer(serializers.Serializer):
    """Serializer para reconocimiento de voz."""
    
    audio = serializers.FileField(required=True)
    use_temporal_smoothing = serializers.BooleanField(default=False)
    
    def validate_audio(self, value):
        """Valida el archivo de audio."""
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        filename = value.name.lower()
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise serializers.ValidationError(
                f"Formato no soportado. Permitidos: {', '.join(allowed_extensions)}"
            )
        
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("El archivo no debe exceder 10MB")
        
        return value


class VoiceVerificationSerializer(serializers.Serializer):
    """Serializer para verificación de voz."""
    
    audio = serializers.FileField(required=True)
    user_id = serializers.IntegerField(required=False)
    username = serializers.CharField(required=False)
    threshold = serializers.FloatField(default=0.5, min_value=0.0, max_value=1.0)
    use_embedding_comparison = serializers.BooleanField(default=False)
    
    def validate(self, data):
        """Valida que se proporcione user_id o username."""
        if not data.get('user_id') and not data.get('username'):
            raise serializers.ValidationError(
                "Debe proporcionar 'user_id' o 'username'"
            )
        
        # Validar que el usuario exista
        if data.get('user_id'):
            if not User.objects.filter(id=data['user_id']).exists():
                raise serializers.ValidationError({'user_id': 'Usuario no encontrado'})
        elif data.get('username'):
            if not User.objects.filter(username=data['username']).exists():
                raise serializers.ValidationError({'username': 'Usuario no encontrado'})
        
        return data
    
    def validate_audio(self, value):
        """Valida el archivo de audio."""
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        filename = value.name.lower()
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            raise serializers.ValidationError(
                f"Formato no soportado. Permitidos: {', '.join(allowed_extensions)}"
            )
        
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("El archivo no debe exceder 10MB")
        
        return value


class VoiceTrainSerializer(serializers.Serializer):
    """Serializer para entrenamiento del modelo."""
    
    epochs = serializers.IntegerField(default=50, min_value=1, max_value=200)
    batch_size = serializers.IntegerField(default=32, min_value=1, max_value=128)
    version = serializers.CharField(default='1.0', max_length=10)
    min_samples_per_user = serializers.IntegerField(default=5, min_value=3, max_value=50)


class VoiceRecognitionModelSerializer(serializers.ModelSerializer):
    """Serializer para modelos de reconocimiento de voz."""
    
    class Meta:
        model = VoiceRecognitionModel
        fields = [
            'id', 'name', 'version', 'model_file', 'embedding_model_file',
            'accuracy', 'loss', 'val_accuracy', 'val_loss',
            'num_classes', 'total_parameters', 'is_active',
            'trained_at', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

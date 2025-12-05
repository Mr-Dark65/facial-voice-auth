from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.conf import settings
from django.core.files.storage import default_storage
from pathlib import Path
import tempfile
import logging

from .models import VoiceSample, VoiceRecognitionModel
from .serializers import (
    VoiceSampleSerializer,
    VoiceUploadSerializer,
    VoiceRegistrationSerializer,
    VoiceRecognitionSerializer,
    VoiceVerificationSerializer,
    VoiceTrainSerializer,
    VoiceRecognitionModelSerializer
)
from .training.trainer import VoiceRecognitionTrainer
from .training.dataset_manager import VoiceDatasetManager
from .inference.predictor import VoiceRecognitionPredictor
from app.users.models import User

logger = logging.getLogger(__name__)


class VoiceSampleViewSet(viewsets.ModelViewSet):
    """ViewSet para gestionar muestras de voz."""
    
    queryset = VoiceSample.objects.all()
    serializer_class = VoiceSampleSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filtrar muestras por usuario si se proporciona."""
        queryset = super().get_queryset()
        user_id = self.request.query_params.get('user_id')
        
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        
        return queryset.order_by('-created_at')


class VoiceRecognitionViewSet(viewsets.ViewSet):
    """ViewSet para funciones de reconocimiento de voz."""
    
    permission_classes = [AllowAny]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_manager = VoiceDatasetManager()
        self.predictor = None
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        """Inicializa el predictor con el modelo activo."""
        try:
            active_model = VoiceRecognitionModel.objects.filter(is_active=True).first()
            
            if active_model and active_model.model_file:
                model_path = Path(settings.MODELS_STORAGE_DIR) / active_model.model_file
                embedding_path = None
                
                if active_model.embedding_model_file:
                    embedding_path = Path(settings.MODELS_STORAGE_DIR) / active_model.embedding_model_file
                
                if model_path.exists():
                    self.predictor = VoiceRecognitionPredictor(
                        model_path=str(model_path),
                        embedding_model_path=str(embedding_path) if embedding_path else None
                    )
                    logger.info(f"Predictor inicializado con modelo: {active_model.name}")
        except Exception as e:
            logger.error(f"Error inicializando predictor: {e}")
            self.predictor = None
    
    def _save_temp_audio(self, audio_file):
        """Guarda archivo de audio temporalmente."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            return temp_file.name
    
    @action(detail=False, methods=['post'], url_path='upload')
    def upload_audio(self, request):
        """
        Sube un archivo de audio sin procesarlo.
        POST /voice/upload/
        """
        serializer = VoiceUploadSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        audio_file = serializer.validated_data['audio']
        user_id = serializer.validated_data.get('user_id')
        
        try:
            # Guardar audio
            audio_path = default_storage.save(
                f'voice_samples/{audio_file.name}',
                audio_file
            )
            
            result = {
                'message': 'Audio subido exitosamente',
                'audio_path': audio_path,
                'filename': audio_file.name,
                'size': audio_file.size
            }
            
            if user_id:
                result['user_id'] = user_id
            
            return Response(result, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error subiendo audio: {e}")
            return Response(
                {'error': f'Error subiendo audio: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'], url_path='register')
    def register_voice(self, request):
        """
        Registra voz de un usuario (guarda audio y extrae embedding).
        POST /voice/register/
        """
        serializer = VoiceRegistrationSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        audio_file = serializer.validated_data['audio']
        user_id = serializer.validated_data['user_id']
        
        try:
            user = User.objects.get(id=user_id)
            
            # Guardar audio temporalmente
            temp_path = self._save_temp_audio(audio_file)
            
            # Cargar audio como array numpy
            import librosa
            audio_data, sr = librosa.load(temp_path, sr=16000)
            
            # Guardar en dataset
            audio_path = self.dataset_manager.save_voice_sample(
                user_id=user_id,
                audio_data=audio_data,
                sample_rate=sr
            )
            
            # Metadata básica
            metadata = {
                'duration': len(audio_data) / sr,
                'sample_rate': sr,
                'snr': None,
                'quality_score': 0.8
            }
            
            # Extraer embedding si hay modelo activo
            embedding = None
            if self.predictor and self.predictor.embedding_model:
                embedding = self.predictor.extract_embedding(audio_path=audio_path)
                embedding_list = embedding.tolist()
            else:
                embedding_list = None
            
            # Crear registro en BD
            voice_sample = VoiceSample.objects.create(
                user=user,
                audio_file=audio_path,
                embedding=embedding_list,
                duration=metadata.get('duration'),
                sample_rate=metadata.get('sample_rate'),
                snr=metadata.get('snr'),
                quality_score=metadata.get('quality_score', 0.0)
            )
            
            # Limpiar archivo temporal
            Path(temp_path).unlink(missing_ok=True)
            
            # Actualizar estado del usuario
            voice_count = VoiceSample.objects.filter(user=user).count()
            user.voice_samples_count = voice_count
            if voice_count >= 3 and not user.voice_registered:
                user.voice_registered = True
            user.save()
            
            return Response({
                'message': 'Voz registrada exitosamente',
                'voice_sample_id': voice_sample.id,
                'user_id': user_id,
                'username': user.username,
                'audio_path': audio_path,
                'metadata': metadata,
                'embedding_extracted': embedding_list is not None,
                'voice_registered': user.voice_registered,
                'total_samples': voice_count
            }, status=status.HTTP_201_CREATED)
            
        except User.DoesNotExist:
            return Response(
                {'error': 'Usuario no encontrado'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error registrando voz: {e}", exc_info=True)
            return Response(
                {'error': f'Error registrando voz: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'], url_path='recognize')
    def recognize_voice(self, request):
        """
        Reconoce la voz y retorna el usuario identificado.
        POST /voice/recognize/
        """
        if not self.predictor:
            return Response(
                {'error': 'Modelo no disponible. Entrena un modelo primero.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        serializer = VoiceRecognitionSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        audio_file = serializer.validated_data['audio']
        use_smoothing = serializer.validated_data['use_temporal_smoothing']
        
        try:
            # Guardar temporalmente
            temp_path = self._save_temp_audio(audio_file)
            
            # Reconocer
            result = self.predictor.recognize_voice(
                audio_path=temp_path,
                use_temporal_smoothing=use_smoothing
            )
            
            # Limpiar
            Path(temp_path).unlink(missing_ok=True)
            
            return Response({
                'message': 'Reconocimiento completado',
                'result': result
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error reconociendo voz: {e}", exc_info=True)
            return Response(
                {'error': f'Error reconociendo voz: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'], url_path='verify')
    def verify_voice(self, request):
        """
        Verifica si el audio pertenece al usuario especificado.
        POST /voice/verify/
        """
        if not self.predictor:
            return Response(
                {'error': 'Modelo no disponible. Entrena un modelo primero.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        serializer = VoiceVerificationSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        audio_file = serializer.validated_data['audio']
        user_id = serializer.validated_data.get('user_id')
        username = serializer.validated_data.get('username')
        threshold = serializer.validated_data['threshold']
        use_embedding = serializer.validated_data['use_embedding_comparison']
        
        try:
            # Obtener usuario
            if user_id:
                user = User.objects.get(id=user_id)
            else:
                user = User.objects.get(username=username)
            
            # Guardar temporalmente
            temp_path = self._save_temp_audio(audio_file)
            
            # Verificar
            if use_embedding and hasattr(user, 'voice_embedding') and user.voice_embedding:
                # Verificación por embedding
                import numpy as np
                stored_embedding = np.array(user.voice_embedding)
                
                result = self.predictor.verify_voice_with_embedding(
                    audio_path=temp_path,
                    stored_embedding=stored_embedding,
                    threshold=threshold
                )
            else:
                # Verificación por clasificación
                result = self.predictor.verify_voice(
                    audio_path=temp_path,
                    user_id=user.id,
                    threshold=threshold
                )
            
            # Limpiar
            Path(temp_path).unlink(missing_ok=True)
            
            return Response({
                'message': 'Verificación completada',
                'result': result
            }, status=status.HTTP_200_OK)
            
        except User.DoesNotExist:
            return Response(
                {'error': 'Usuario no encontrado'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error verificando voz: {e}", exc_info=True)
            return Response(
                {'error': f'Error verificando voz: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'], url_path='train')
    def train_model(self, request):
        """
        Entrena un nuevo modelo de reconocimiento de voz.
        POST /voice/train/
        """
        serializer = VoiceTrainSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        epochs = serializer.validated_data['epochs']
        batch_size = serializer.validated_data['batch_size']
        version = serializer.validated_data['version']
        min_samples = serializer.validated_data['min_samples_per_user']
        
        try:
            # Crear trainer
            trainer = VoiceRecognitionTrainer()
            
            # Entrenar
            logger.info(f"Iniciando entrenamiento - epochs={epochs}, batch_size={batch_size}")
            model_info = trainer.full_training_pipeline(
                epochs=epochs,
                batch_size=batch_size,
                version=version
            )
            
            # Crear registro en BD
            metadata = model_info['metadata']
            training_info = metadata.get('training', {})
            
            voice_model = VoiceRecognitionModel.objects.create(
                name=metadata['model_name'],
                version=version,
                model_file=Path(model_info['model_path']).name,
                embedding_model_file=Path(model_info['embedding_path']).name,
                accuracy=training_info.get('final_accuracy', 0),
                loss=training_info.get('final_loss', 0),
                val_accuracy=training_info.get('final_val_accuracy', 0),
                val_loss=training_info.get('final_val_loss', 0),
                num_classes=metadata.get('num_classes', 0),
                total_parameters=metadata.get('total_parameters', 0),
                is_active=True
            )
            
            # Desactivar otros modelos
            VoiceRecognitionModel.objects.exclude(id=voice_model.id).update(is_active=False)
            
            # Reinicializar predictor con nuevo modelo
            self._initialize_predictor()
            
            return Response({
                'message': 'Modelo entrenado exitosamente',
                'model_id': voice_model.id,
                'model_info': model_info,
                'training_metrics': training_info
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}", exc_info=True)
            return Response(
                {'error': f'Error entrenando modelo: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'], url_path='model-info')
    def model_info(self, request):
        """
        Obtiene información del modelo activo.
        GET /voice/model-info/
        """
        if not self.predictor:
            return Response(
                {'error': 'No hay modelo activo'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        info = self.predictor.get_model_info()
        
        return Response({
            'message': 'Información del modelo',
            'info': info
        }, status=status.HTTP_200_OK)


class VoiceRecognitionModelViewSet(viewsets.ModelViewSet):
    """ViewSet para gestionar modelos de reconocimiento de voz."""
    
    queryset = VoiceRecognitionModel.objects.all()
    serializer_class = VoiceRecognitionModelSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Ordenar por fecha de entrenamiento."""
        return super().get_queryset().order_by('-trained_at')
    
    @action(detail=True, methods=['post'], url_path='activate')
    def activate_model(self, request, pk=None):
        """
        Activa un modelo específico.
        POST /voice/models/{id}/activate/
        """
        model = self.get_object()
        
        # Desactivar todos
        VoiceRecognitionModel.objects.update(is_active=False)
        
        # Activar este
        model.is_active = True
        model.save()
        
        return Response({
            'message': f'Modelo {model.name} activado',
            'model_id': model.id
        }, status=status.HTTP_200_OK)

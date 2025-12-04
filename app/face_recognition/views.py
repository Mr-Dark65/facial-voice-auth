from django.views.generic import TemplateView
from django.shortcuts import render
from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from django.utils import timezone
from PIL import Image
import numpy as np
import cv2
import io
import json
import logging

from .models import FaceImage, FaceRecognitionModel
from .serializers import (
    FaceImageSerializer, FaceImageUploadSerializer, FaceRegistrationSerializer,
    FaceVerificationSerializer, FaceRecognitionSerializer,
    FaceRecognitionModelSerializer, TrainModelSerializer,
    DatasetStatsSerializer, FaceEmbeddingSerializer, FaceRegistrationVideoSerializer
)
from .preprocessing.detector import FaceDetector
from .preprocessing.transformer import FaceTransformer
from .training.dataset_manager import FaceDatasetManager
from .training.trainer import FaceRecognitionTrainer
from .inference.predictor import FaceRecognitionPredictor
from app.users.models import User
import tempfile
from django.core.files.base import ContentFile
import os

logger = logging.getLogger(__name__)
class RegisterFaceFromVideoView(APIView):
    """
    Vista para registrar imágenes faciales a partir de un video.
    POST /api/face/register-video/
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = FaceRegistrationVideoSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        video_file = serializer.validated_data['video']
        detector = FaceDetector()
        dataset_manager = FaceDatasetManager()

        # Guardar el video temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(int(fps), 5)  # Extraer 1 frame por segundo o cada 5 frames mínimo

        successful_uploads = []
        failed_uploads = []
        idx = 0
        saved_images = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval != 0:
                idx += 1
                continue

            # Convertir a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            validation = detector.validate_face_quality(image_array=frame_rgb)

            if not validation['is_valid']:
                failed_uploads.append({
                    'frame': idx,
                    'reason': validation['message']
                })
                idx += 1
                continue

            face = detector.extract_face(image_array=frame_rgb)
            if face is None:
                failed_uploads.append({
                    'frame': idx,
                    'reason': 'No se pudo extraer el rostro'
                })
                idx += 1
                continue

            # Guardar en base de datos
            # Convertir el frame a imagen JPEG en memoria
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            if not is_success:
                failed_uploads.append({
                    'frame': idx,
                    'reason': 'No se pudo codificar la imagen'
                })
                idx += 1
                continue
            image_content = ContentFile(buffer.tobytes(), name=f"video_face_{request.user.id}_{idx}.jpg")

            face_image = FaceImage.objects.create(
                user=request.user,
                image=image_content,
                is_valid=True,
                quality_score=validation['quality_score'],
                face_detected=True,
                processed_at=timezone.now()
            )

            dataset_manager.save_face_image(
                user_id=request.user.id,
                image=face,
                image_name=f"face_{face_image.id}.jpg"
            )

            successful_uploads.append({
                'frame': idx,
                'id': face_image.id,
                'quality_score': validation['quality_score']
            })
            saved_images += 1
            idx += 1

        cap.release()
        os.remove(temp_video_path)

        # Actualizar usuario
        total_images = FaceImage.objects.filter(
            user=request.user,
            is_valid=True
        ).count()
        request.user.face_images_count = total_images
        if total_images >= settings.MIN_FACE_IMAGES:
            request.user.face_registered = True
        request.user.save()

        return Response({
            'message': f'{saved_images} imágenes extraídas y registradas exitosamente',
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'total_images': total_images,
            'face_registered': request.user.face_registered
        }, status=status.HTTP_201_CREATED)
        
class FaceVerificationVideoView(TemplateView):
    template_name = 'face_verification_video.html'

class UploadFaceImageView(APIView):
    """
    Vista para subir una imagen facial individual.
    POST /api/face/upload/
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        serializer = FaceImageUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        image_file = serializer.validated_data['image']
        
        # Convertir a array de numpy
        image_bytes = image_file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image_pil)
        
        # Convertir a RGB si es necesario
        if len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Detectar y validar rostro
        detector = FaceDetector()
        validation = detector.validate_face_quality(image_array=image_array)
        
        if not validation['is_valid']:
            return Response({
                'error': validation['message'],
                'quality_score': validation['quality_score']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Extraer rostro
        face = detector.extract_face(image_array=image_array)
        
        if face is None:
            return Response({
                'error': 'No se pudo extraer el rostro'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Crear registro de imagen
        face_image = FaceImage.objects.create(
            user=request.user,
            image=image_file,
            is_valid=True,
            quality_score=validation['quality_score'],
            face_detected=True,
            processed_at=timezone.now()
        )
        
        # Guardar imagen en dataset
        dataset_manager = FaceDatasetManager()
        dataset_manager.save_face_image(
            user_id=request.user.id,
            image=face,
            image_name=f"face_{face_image.id}.jpg"
        )
        
        # Actualizar contador del usuario
        total_images = FaceImage.objects.filter(
            user=request.user,
            is_valid=True
        ).count()
        
        request.user.face_images_count = total_images
        
        # Verificar si alcanzó el mínimo para registro
        min_images = settings.MIN_FACE_IMAGES
        if total_images >= min_images:
            request.user.face_registered = True
        
        request.user.save()
        
        return Response({
            'message': 'Imagen facial subida exitosamente',
            'face_image': FaceImageSerializer(face_image).data,
            'total_images': total_images,
            'required_images': min_images,
            'face_registered': request.user.face_registered
        }, status=status.HTTP_201_CREATED)


class RegisterFaceView(APIView):
    """
    Vista para registrar múltiples imágenes faciales de un usuario.
    POST /api/face/register/
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        serializer = FaceRegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        images = serializer.validated_data['images']
        
        detector = FaceDetector()
        transformer = FaceTransformer()
        dataset_manager = FaceDatasetManager()
        
        successful_uploads = []
        failed_uploads = []
        
        for idx, image_file in enumerate(images):
            try:
                # Convertir a array
                image_bytes = image_file.read()
                image_pil = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image_pil)
                
                # Convertir a RGB
                if len(image_array.shape) == 2:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                elif image_array.shape[2] == 4:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                
                # Validar calidad
                validation = detector.validate_face_quality(image_array=image_array)
                
                if not validation['is_valid']:
                    failed_uploads.append({
                        'index': idx,
                        'reason': validation['message']
                    })
                    continue
                
                # Extraer rostro
                face = detector.extract_face(image_array=image_array)
                
                if face is None:
                    failed_uploads.append({
                        'index': idx,
                        'reason': 'No se pudo extraer el rostro'
                    })
                    continue
                
                # Guardar en base de datos
                face_image = FaceImage.objects.create(
                    user=request.user,
                    image=image_file,
                    is_valid=True,
                    quality_score=validation['quality_score'],
                    face_detected=True,
                    processed_at=timezone.now()
                )
                
                # Guardar en dataset
                dataset_manager.save_face_image(
                    user_id=request.user.id,
                    image=face,
                    image_name=f"face_{face_image.id}.jpg"
                )
                
                successful_uploads.append({
                    'index': idx,
                    'id': face_image.id,
                    'quality_score': validation['quality_score']
                })
                
            except Exception as e:
                logger.error(f"Error procesando imagen {idx}: {str(e)}")
                failed_uploads.append({
                    'index': idx,
                    'reason': str(e)
                })
        
        # Actualizar usuario
        total_images = FaceImage.objects.filter(
            user=request.user,
            is_valid=True
        ).count()
        
        request.user.face_images_count = total_images
        
        if total_images >= settings.MIN_FACE_IMAGES:
            request.user.face_registered = True
        
        request.user.save()
        
        return Response({
            'message': f'{len(successful_uploads)} imágenes procesadas exitosamente',
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'total_images': total_images,
            'face_registered': request.user.face_registered
        }, status=status.HTTP_201_CREATED)


class FaceImagesListView(generics.ListAPIView):
    """
    Vista para listar las imágenes faciales del usuario.
    GET /api/face/my-images/
    """
    serializer_class = FaceImageSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return FaceImage.objects.filter(user=self.request.user)


class DeleteFaceImageView(generics.DestroyAPIView):
    """
    Vista para eliminar una imagen facial.
    DELETE /api/face/images/<id>/
    """
    queryset = FaceImage.objects.all()
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return FaceImage.objects.filter(user=self.request.user)
    
    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        
        # Actualizar contador del usuario
        instance.user.face_images_count = FaceImage.objects.filter(
            user=instance.user,
            is_valid=True
        ).count() - 1
        
        # Verificar si aún cumple con el mínimo
        if instance.user.face_images_count < settings.MIN_FACE_IMAGES:
            instance.user.face_registered = False
        
        instance.user.save()
        
        self.perform_destroy(instance)
        
        return Response({
            'message': 'Imagen eliminada exitosamente',
            'remaining_images': instance.user.face_images_count
        }, status=status.HTTP_200_OK)


class TrainFaceModelView(APIView):
    """
    Vista para entrenar un nuevo modelo de reconocimiento facial.
    POST /api/face/train/
    """
    permission_classes = [permissions.IsAdminUser]
    
    def post(self, request):
        serializer = TrainModelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        epochs = serializer.validated_data['epochs']
        batch_size = serializer.validated_data['batch_size']
        version = serializer.validated_data['version']
        
        try:
            # Crear trainer
            trainer = FaceRecognitionTrainer()
            
            # Validar dataset
            validation = trainer.dataset_manager.validate_dataset()
            
            if not validation['is_valid']:
                return Response({
                    'error': 'Dataset no válido para entrenamiento',
                    'issues': validation['issues'],
                    'statistics': validation['statistics']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Entrenar
            logger.info(f"Iniciando entrenamiento con {epochs} épocas...")
            model_info = trainer.full_training_pipeline(
                epochs=epochs,
                batch_size=batch_size,
                version=version
            )
            
            # Guardar en base de datos
            metadata = model_info['metadata']
            training_info = metadata.get('training', {})
            
            # Eliminar modelo anterior con el mismo nombre (si existe)
            FaceRecognitionModel.objects.filter(name=metadata['model_name']).delete()
            face_model = FaceRecognitionModel.objects.create(
                name=metadata['model_name'],
                version=version,
                model_file=model_info['model_path'],
                accuracy=training_info.get('final_accuracy'),
                loss=training_info.get('final_loss'),
                val_accuracy=training_info.get('final_val_accuracy'),
                val_loss=training_info.get('final_val_loss'),
                training_samples=len(trainer.dataset_manager.get_all_users_with_images()),
                epochs_trained=training_info.get('epochs_trained', epochs),
                architecture=json.dumps(metadata),
                is_active=True,
                trained_at=timezone.now()
            )
            
            return Response({
                'message': 'Modelo entrenado exitosamente',
                'model': FaceRecognitionModelSerializer(face_model).data,
                'training_info': training_info
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {str(e)}")
            return Response({
                'error': f'Error durante el entrenamiento: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DatasetStatsView(APIView):
    """
    Vista para obtener estadísticas del dataset.
    GET /api/face/dataset-stats/
    """
    permission_classes = [permissions.IsAdminUser]
    
    def get(self, request):
        dataset_manager = FaceDatasetManager()
        stats = dataset_manager.get_dataset_statistics()
        
        # Agregar campo is_ready_for_training
        stats['is_ready_for_training'] = (
            stats['total_users'] >= 2 and
            stats['min_images_per_user'] >= settings.MIN_FACE_IMAGES
        )
        
        serializer = DatasetStatsSerializer(stats)
        return Response(serializer.data)


@method_decorator(csrf_exempt, name='dispatch')
class FaceRecognitionView(APIView):
    """
    Vista para reconocer un rostro (identificación).
    POST /api/face/recognize/
    """
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        import logging
        logger = logging.getLogger('django')
        logger.debug(f"[DEBUG] request.data: {request.data}")
        logger.debug(f"[DEBUG] request.FILES: {request.FILES}")
        try:
            serializer = FaceRecognitionSerializer(data=request.data)
            if not serializer.is_valid():
                logger.error(f"[DEBUG] serializer.errors: {serializer.errors}")
                return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
            image_file = serializer.validated_data['image']
        except Exception as e:
            logger.error(f"[DEBUG] Error en validación inicial: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        try:
            # Convertir imagen
            image_bytes = image_file.read()
            image_pil = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image_pil)
            
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Cargar predictor
            predictor = FaceRecognitionPredictor()
            predictor.load_latest_model()
            
            # Reconocer
            result = predictor.recognize_face_from_image(image_array=image_array)
            
            if not result['success']:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)

            # Obtener información del usuario
            user_data = None
            if result['user_id']:
                try:
                    user = User.objects.get(id=result['user_id'])
                    user_data = {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'full_name': user.get_full_name()
                    }
                except User.DoesNotExist:
                    user_data = None

            response_data = {**result}
            if user_data:
                response_data['user'] = user_data
            else:
                response_data['user'] = None

            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error en reconocimiento: {str(e)}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FaceVerificationView(APIView):
    """
    Vista para verificar un rostro contra un usuario específico.
    POST /api/face/verify/
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        serializer = FaceVerificationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        image_file = serializer.validated_data['image']
        user_id = serializer.validated_data.get('user_id', request.user.id)
        
        try:
            # Obtener usuario
            user = User.objects.get(id=user_id)
            
            if not user.face_embedding:
                return Response({
                    'error': 'El usuario no tiene embedding facial registrado'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Convertir imagen
            image_bytes = image_file.read()
            image_pil = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image_pil)
            
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Cargar predictor
            predictor = FaceRecognitionPredictor()
            predictor.load_latest_model()
            
            # Verificar
            result = predictor.verify_face_from_image(
                image_array=image_array,
                stored_embedding=user.face_embedding
            )
            
            return Response(result)
            
        except User.DoesNotExist:
            return Response({
                'error': 'Usuario no encontrado'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error en verificación: {str(e)}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ModelListView(generics.ListAPIView):
    """
    Vista para listar modelos entrenados.
    GET /api/face/models/
    """
    queryset = FaceRecognitionModel.objects.all()
    serializer_class = FaceRecognitionModelSerializer
    permission_classes = [permissions.IsAdminUser]

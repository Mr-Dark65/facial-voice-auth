import tensorflow as tf
from keras import models
import numpy as np
from pathlib import Path
import json
import logging
from django.conf import settings
from scipy.spatial.distance import cosine, euclidean

from ..preprocessing.detector import FaceDetector
from ..preprocessing.transformer import FaceTransformer

logger = logging.getLogger(__name__)


class FaceRecognitionPredictor:
    """
    Predictor para reconocimiento facial en producción.
    Realiza inferencia con modelos entrenados y compara embeddings.
    """
    
    def __init__(self, model_path=None, embedding_model_path=None, metadata_path=None):

        self.model = None
        self.embedding_model = None
        self.metadata = None
        self.user_id_mapping = None
        self.reverse_user_mapping = None
        
        # Componentes
        self.face_detector = FaceDetector()
        self.face_transformer = FaceTransformer()
        
        # Cargar modelo si se proporciona
        if model_path:
            self.load_model(model_path, embedding_model_path, metadata_path)
        
        logger.info("FaceRecognitionPredictor inicializado")
    
    def load_model(self, model_path, embedding_model_path=None, metadata_path=None):
        logger.info(f"Cargando modelo desde: {model_path}")
        
        # Cargar modelo principal
        self.model = models.load_model(model_path)
        
        # Cargar modelo de embeddings
        if embedding_model_path:
            self.embedding_model = models.load_model(embedding_model_path)
            logger.info(f"Modelo de embeddings cargado desde: {embedding_model_path}")
        
        # Cargar metadata
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Crear mapping de user IDs (convertir keys a int)
            if 'user_id_mapping' in self.metadata:
                self.user_id_mapping = {
                    int(k): v for k, v in self.metadata['user_id_mapping'].items()
                }
                # Crear reverse mapping (de índice de clase a user_id)
                self.reverse_user_mapping = {
                    v: int(k) for k, v in self.user_id_mapping.items()
                }
            
            logger.info("Metadata cargada exitosamente")
        
        logger.info("Modelo cargado exitosamente")
    
    def load_latest_model(self):
        """
        Carga el modelo más reciente del directorio de modelos.
        """
        models_dir = Path(settings.MODELS_STORAGE_DIR)
        
        # Buscar archivos de modelo
        model_files = list(models_dir.glob("face_recognition_model_v*.keras"))
        
        if not model_files:
            raise FileNotFoundError("No se encontraron modelos entrenados")
        
        # Tomar el más reciente
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # Buscar embedding model y metadata correspondientes
        base_name = latest_model.stem.replace('face_recognition_model_', '')
        embedding_model = models_dir / f"face_recognition_model_embedding_{base_name}.keras"
        metadata_file = models_dir / f"face_recognition_model_metadata_{base_name}.json"
        
        # Cargar
        self.load_model(
            str(latest_model),
            str(embedding_model) if embedding_model.exists() else None,
            str(metadata_file) if metadata_file.exists() else None
        )
        
        logger.info(f"Modelo más reciente cargado: {latest_model.name}")
    
    def extract_embedding(self, face_image):
        if self.embedding_model is None:
            raise ValueError("Modelo de embeddings no cargado")
        
        # Asegurar que tenga la dimensión de batch
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        # Extraer embedding
        embedding = self.embedding_model.predict(face_image, verbose=0)
        
        return embedding[0]  # Retornar solo el primer elemento
    
    def predict_class(self, face_image):
        if self.model is None:
            raise ValueError("Modelo no cargado")
        
        # Asegurar dimensión de batch
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        # Predicción
        predictions = self.model.predict(face_image, verbose=0)
        
        # Obtener clase predicha y confianza
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        # Obtener user_id correspondiente
        user_id = None
        if self.reverse_user_mapping:
            user_id = self.reverse_user_mapping.get(class_idx)
        
        return {
            'class_index': int(class_idx),
            'confidence': confidence,
            'user_id': user_id,
            'all_probabilities': predictions[0].tolist()
        }
    
    def recognize_face_from_image(self, image_path=None, image_array=None):
        # Detectar y extraer rostro
        face = self.face_detector.extract_face(
            image_path=image_path,
            image_array=image_array
        )
        
        if face is None:
            return {
                'success': False,
                'message': 'No se detectó rostro en la imagen',
                'user_id': None,
                'confidence': 0.0
            }
        
        # Preprocesar
        face_processed = self.face_transformer.preprocess_for_inference(face)
        
        # Predecir
        prediction = self.predict_class(face_processed)
        
        return {
            'success': True,
            'user_id': prediction['user_id'],
            'confidence': prediction['confidence'],
            'class_index': prediction['class_index'],
            'message': 'Rostro reconocido exitosamente'
        }
    
    def verify_face(self, face_image, stored_embedding, threshold=None):
        if threshold is None:
            threshold = settings.FACE_RECOGNITION_THRESHOLD
        
        # Extraer embedding de la imagen
        current_embedding = self.extract_embedding(face_image)
        
        # Convertir stored_embedding a numpy si es lista
        if isinstance(stored_embedding, list):
            stored_embedding = np.array(stored_embedding)
        
        # Calcular similitud (usando distancia coseno)
        # Menor distancia = mayor similitud
        cosine_distance = cosine(current_embedding, stored_embedding)
        similarity = 1 - cosine_distance  # Convertir a similitud
        
        # Calcular distancia euclidiana (opcional)
        euclidean_dist = euclidean(current_embedding, stored_embedding)
        
        # Verificar si pasa el umbral
        is_match = similarity >= threshold
        
        return {
            'is_match': is_match,
            'similarity': float(similarity),
            'cosine_distance': float(cosine_distance),
            'euclidean_distance': float(euclidean_dist),
            'threshold': threshold,
            'confidence': float(similarity)
        }
    
    def verify_face_from_image(self, image_path=None, image_array=None, stored_embedding=None, threshold=None):
        # Detectar y extraer rostro
        face = self.face_detector.extract_face(
            image_path=image_path,
            image_array=image_array
        )
        
        if face is None:
            return {
                'success': False,
                'is_match': False,
                'message': 'No se detectó rostro en la imagen',
                'confidence': 0.0
            }
        
        # Preprocesar
        face_processed = self.face_transformer.preprocess_for_training(face)
        
        # Verificar
        result = self.verify_face(face_processed, stored_embedding, threshold)
        
        result['success'] = True
        result['message'] = 'Verificación completada'
        
        return result
    
    def compare_faces(self, face1, face2):
        # Extraer embeddings
        embedding1 = self.extract_embedding(face1)
        embedding2 = self.extract_embedding(face2)
        
        # Calcular similitud
        cosine_distance = cosine(embedding1, embedding2)
        similarity = 1 - cosine_distance
        
        return {
            'similarity': float(similarity),
            'cosine_distance': float(cosine_distance),
            'are_same_person': similarity >= settings.FACE_RECOGNITION_THRESHOLD
        }
    
    def batch_predict(self, face_images):
        if self.model is None:
            raise ValueError("Modelo no cargado")
        
        predictions = self.model.predict(face_images, verbose=0)
        
        results = []
        for pred in predictions:
            class_idx = np.argmax(pred)
            confidence = float(pred[class_idx])
            user_id = self.reverse_user_mapping.get(class_idx) if self.reverse_user_mapping else None
            
            results.append({
                'class_index': int(class_idx),
                'confidence': confidence,
                'user_id': user_id
            })
        
        return results

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from django.conf import settings
from typing import Optional, Dict, Tuple
import json
import logging
from collections import deque

from ..preprocessing.mfcc_extractor import MFCCExtractor
from ..preprocessing.audio_cleaner import AudioCleaner
from app.users.models import User

logger = logging.getLogger(__name__)


class VoiceRecognitionPredictor:
    """
    Predictor para reconocimiento y verificación de voz.
    Carga modelos entrenados y realiza predicciones.
    """
    
    def __init__(self, model_path: Optional[str] = None, embedding_model_path: Optional[str] = None):
        self.model = None
        self.embedding_model = None
        self.metadata = None
        self.user_id_mapping = None
        self.reverse_mapping = None
        
        # Componentes
        self.mfcc_extractor = MFCCExtractor()
        self.audio_cleaner = AudioCleaner()
        
        # Temporal smoothing para WebSocket
        self.temporal_predictions = deque(maxlen=5)
        
        # Cargar modelos si se proporcionan paths
        if model_path:
            self.load_model(model_path, embedding_model_path)
    
    def load_model(self, model_path: str, embedding_model_path: Optional[str] = None):
        """Carga modelo entrenado y metadata."""
        logger.info(f"Cargando modelo desde: {model_path}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Cargar modelo principal
        self.model = keras.models.load_model(str(model_path))
        logger.info("Modelo principal cargado")
        
        # Cargar modelo de embeddings
        if embedding_model_path:
            embedding_path = Path(embedding_model_path)
            if embedding_path.exists():
                self.embedding_model = keras.models.load_model(str(embedding_path))
                logger.info("Modelo de embeddings cargado")
        
        # Cargar metadata
        metadata_path = model_path.parent / model_path.name.replace('.keras', '_metadata.json').replace('_v', '_metadata_v')
        
        # Intentar variaciones del nombre
        if not metadata_path.exists():
            # Buscar cualquier metadata con timestamp similar
            pattern = model_path.stem.split('_v')[0] + '_metadata_*.json'
            metadata_files = list(model_path.parent.glob(pattern))
            if metadata_files:
                metadata_path = sorted(metadata_files)[-1]  # Más reciente
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.user_id_mapping = self.metadata.get('user_id_mapping', {})
            self.reverse_mapping = {v: k for k, v in self.user_id_mapping.items()}
            
            logger.info(f"Metadata cargada: {len(self.user_id_mapping)} usuarios")
        else:
            logger.warning(f"Metadata no encontrada: {metadata_path}")
    
    def preprocess_audio(self, audio_path: Optional[str] = None, 
                        audio_array: Optional[np.ndarray] = None,
                        target_timesteps: int = 100) -> np.ndarray:
        """Preprocesa audio y extrae características MFCC."""
        
        # Cargar y limpiar audio
        if audio_path:
            audio_clean = self.audio_cleaner.preprocess_audio(audio_path=audio_path)
        elif audio_array is not None:
            audio_clean = self.audio_cleaner.preprocess_audio(audio_array=audio_array)
        else:
            raise ValueError("Debe proporcionar audio_path o audio_array")
        
        # Extraer MFCCs
        mfccs = self.mfcc_extractor.extract_mfcc(
            audio_array=audio_clean,
            delta=True,
            delta_delta=True
        )
        
        if mfccs is None:
            raise ValueError("No se pudieron extraer características MFCC")
        
        # Normalizar
        mfccs = self.mfcc_extractor.normalize_features(mfccs)
        
        # Ajustar longitud
        mfccs = self.mfcc_extractor.pad_or_truncate(mfccs, target_timesteps)
        
        # Transponer para (timesteps, features) y expandir dimensión batch
        mfccs = mfccs.T
        mfccs = np.expand_dims(mfccs, axis=0)
        
        return mfccs
    
    def extract_embedding(self, audio_path: Optional[str] = None,
                         audio_array: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae embedding de voz normalizado."""
        if self.embedding_model is None:
            raise ValueError("Modelo de embeddings no cargado")
        
        # Preprocesar
        features = self.preprocess_audio(audio_path, audio_array)
        
        # Extraer embedding
        embedding = self.embedding_model.predict(features, verbose=0)[0]
        
        return embedding
    
    def recognize_voice(self, audio_path: Optional[str] = None,
                       audio_array: Optional[np.ndarray] = None,
                       use_temporal_smoothing: bool = False) -> Dict:
        """
        Reconoce voz y retorna predicción con confianza.
        
        Returns:
            Dict con: user_id, username, confidence, all_probabilities
        """
        if self.model is None:
            raise ValueError("Modelo no cargado")
        
        # Preprocesar
        features = self.preprocess_audio(audio_path, audio_array)
        
        # Predicción
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Temporal smoothing
        if use_temporal_smoothing:
            self.temporal_predictions.append(predictions)
            predictions = self._smooth_predictions()
        
        # Mejor predicción
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])
        
        # Mapear a user_id
        user_id = self.reverse_mapping.get(predicted_class)
        
        # Obtener username
        username = None
        if user_id:
            try:
                user = User.objects.get(id=user_id)
                username = user.username
            except User.DoesNotExist:
                logger.warning(f"Usuario {user_id} no encontrado")
        
        result = {
            'user_id': user_id,
            'username': username,
            'confidence': confidence,
            'predicted_class': predicted_class,
            'all_probabilities': {
                self.reverse_mapping.get(i, f'class_{i}'): float(prob)
                for i, prob in enumerate(predictions)
            }
        }
        
        return result
    
    def verify_voice(self, audio_path: Optional[str] = None,
                    audio_array: Optional[np.ndarray] = None,
                    user_id: Optional[int] = None,
                    username: Optional[str] = None,
                    threshold: float = 0.5) -> Dict:
        """
        Verifica si el audio pertenece al usuario especificado.
        
        Args:
            audio_path: Ruta al archivo de audio
            audio_array: Array de audio
            user_id: ID del usuario a verificar
            username: Username del usuario (alternativa a user_id)
            threshold: Umbral de confianza (0.5 = 50%)
        
        Returns:
            Dict con: verified, confidence, user_id, username
        """
        
        # Obtener user_id si se proporciona username
        if username and not user_id:
            try:
                user = User.objects.get(username=username)
                user_id = user.id
            except User.DoesNotExist:
                return {
                    'verified': False,
                    'confidence': 0.0,
                    'user_id': None,
                    'username': username,
                    'error': 'Usuario no encontrado'
                }
        
        # Reconocer voz
        recognition = self.recognize_voice(audio_path, audio_array)
        
        # Verificar si coincide
        verified = (recognition['user_id'] == user_id and 
                   recognition['confidence'] >= threshold)
        
        return {
            'verified': verified,
            'confidence': recognition['confidence'],
            'user_id': recognition['user_id'],
            'username': recognition['username'],
            'expected_user_id': user_id,
            'threshold': threshold
        }
    
    def verify_voice_with_embedding(self, audio_path: Optional[str] = None,
                                   audio_array: Optional[np.ndarray] = None,
                                   stored_embedding: np.ndarray = None,
                                   threshold: float = 0.75) -> Dict:
        """
        Verifica voz comparando embeddings directamente (más preciso).
        
        Args:
            audio_path: Ruta al archivo de audio
            audio_array: Array de audio
            stored_embedding: Embedding almacenado del usuario
            threshold: Umbral de similitud coseno (0.75 = 75%)
        
        Returns:
            Dict con: verified, similarity, threshold
        """
        if stored_embedding is None:
            raise ValueError("stored_embedding es requerido")
        
        # Extraer embedding del audio
        new_embedding = self.extract_embedding(audio_path, audio_array)
        
        # Calcular similitud coseno (embeddings ya están normalizados)
        similarity = float(np.dot(new_embedding, stored_embedding))
        
        # Verificar
        verified = similarity >= threshold
        
        return {
            'verified': verified,
            'similarity': similarity,
            'threshold': threshold
        }
    
    def _smooth_predictions(self) -> np.ndarray:
        """Suaviza predicciones usando promedio ponderado temporal."""
        if len(self.temporal_predictions) == 0:
            return None
        
        # Promedio ponderado (más peso a predicciones recientes)
        weights = np.linspace(0.5, 1.0, len(self.temporal_predictions))
        weights = weights / weights.sum()
        
        smoothed = np.average(
            list(self.temporal_predictions),
            weights=weights,
            axis=0
        )
        
        return smoothed
    
    def clear_temporal_smoothing(self):
        """Limpia historial de predicciones temporales."""
        self.temporal_predictions.clear()
    
    def get_model_info(self) -> Dict:
        """Retorna información del modelo cargado."""
        info = {
            'model_loaded': self.model is not None,
            'embedding_model_loaded': self.embedding_model is not None,
            'num_users': len(self.user_id_mapping) if self.user_id_mapping else 0
        }
        
        if self.metadata:
            info.update({
                'model_name': self.metadata.get('model_name'),
                'version': self.metadata.get('version'),
                'timestamp': self.metadata.get('timestamp'),
                'training_info': self.metadata.get('training')
            })
        
        return info

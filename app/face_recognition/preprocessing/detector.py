import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detector de rostros usando MTCNN (Multi-task Cascaded Convolutional Networks).
    Detecta rostros y puntos faciales clave en imágenes.
    """
    
    def __init__(self, min_confidence=0.9):
        """
        Inicializa el detector de rostros.
        
        Args:
            min_confidence (float): Confianza mínima para aceptar una detección (0-1)
        """
        self.detector = MTCNN()
        self.min_confidence = min_confidence
        logger.info(f"FaceDetector inicializado con confianza mínima: {min_confidence}")
    
    def detect_face(self, image_path=None, image_array=None):
        """
        Detecta rostros en una imagen.
        
        Args:
            image_path (str): Ruta a la imagen
            image_array (np.array): Array numpy de la imagen
        
        Returns:
            dict: Información de la detección o None si no se detecta rostro
        """
        try:
            # Cargar imagen
            if image_path:
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"No se pudo cargar la imagen: {image_path}")
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image_array is not None:
                image = image_array
            else:
                logger.error("Debe proporcionar image_path o image_array")
                return None
            
            # Detectar rostros
            detections = self.detector.detect_faces(image)
            
            if not detections:
                logger.warning("No se detectaron rostros en la imagen")
                return None
            
            # Tomar la detección con mayor confianza
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            if best_detection['confidence'] < self.min_confidence:
                logger.warning(
                    f"Confianza de detección ({best_detection['confidence']:.2f}) "
                    f"menor que el mínimo ({self.min_confidence})"
                )
                return None
            
            return {
                'box': best_detection['box'],  # [x, y, width, height]
                'confidence': best_detection['confidence'],
                'keypoints': best_detection['keypoints'],
                'total_faces': len(detections)
            }
            
        except Exception as e:
            logger.error(f"Error detectando rostro: {str(e)}")
            return None
    
    def extract_face(self, image_path=None, image_array=None, target_size=(160, 160)):
        """
        Extrae y recorta el rostro de una imagen.
        
        Args:
            image_path (str): Ruta a la imagen
            image_array (np.array): Array numpy de la imagen
            target_size (tuple): Tamaño objetivo para el rostro extraído
        
        Returns:
            np.array: Imagen del rostro recortado y redimensionado, o None
        """
        try:
            # Cargar imagen
            if image_path:
                image = cv2.imread(image_path)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image_array is not None:
                image = image_array
            else:
                return None
            
            # Detectar rostro
            detection = self.detect_face(image_array=image)
            
            if detection is None:
                return None
            
            # Extraer coordenadas del rostro
            x, y, width, height = detection['box']
            
            # Asegurar que las coordenadas estén dentro de los límites de la imagen
            x = max(0, x)
            y = max(0, y)
            x2 = min(image.shape[1], x + width)
            y2 = min(image.shape[0], y + height)
            
            # Recortar rostro
            face = image[y:y2, x:x2]
            
            # Redimensionar al tamaño objetivo
            face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error extrayendo rostro: {str(e)}")
            return None
    
    def validate_face_quality(self, image_path=None, image_array=None):
        """
        Valida la calidad de una imagen facial.
        
        Args:
            image_path (str): Ruta a la imagen
            image_array (np.array): Array numpy de la imagen
        
        Returns:
            dict: Información de validación con score de calidad
        """
        detection = self.detect_face(image_path=image_path, image_array=image_array)
        
        if detection is None:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'message': 'No se detectó rostro o confianza muy baja'
            }
        
        # Calcular score de calidad basado en varios factores
        quality_score = detection['confidence']
        
        # Penalizar si se detectan múltiples rostros
        if detection['total_faces'] > 1:
            quality_score *= 0.8
            message = f"Se detectaron {detection['total_faces']} rostros. Use una imagen con un solo rostro."
        else:
            message = "Imagen válida"
        
        # Verificar tamaño del rostro detectado
        box = detection['box']
        face_area = box[2] * box[3]
        
        if face_area < 10000:  # Rostro muy pequeño
            quality_score *= 0.7
            message = "Rostro detectado es muy pequeño. Acérquese más."
        
        is_valid = quality_score >= self.min_confidence * 0.9
        
        return {
            'is_valid': is_valid,
            'quality_score': float(quality_score),
            'confidence': float(detection['confidence']),
            'total_faces': detection['total_faces'],
            'message': message
        }
    
    def detect_multiple_faces(self, image_path=None, image_array=None):
        """
        Detecta todos los rostros en una imagen.
        
        Args:
            image_path (str): Ruta a la imagen
            image_array (np.array): Array numpy de la imagen
        
        Returns:
            list: Lista de detecciones de rostros
        """
        try:
            # Cargar imagen
            if image_path:
                image = cv2.imread(image_path)
                if image is None:
                    return []
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image_array is not None:
                image = image_array
            else:
                return []
            
            # Detectar todos los rostros
            detections = self.detector.detect_faces(image)
            
            # Filtrar por confianza mínima
            valid_detections = [
                d for d in detections 
                if d['confidence'] >= self.min_confidence
            ]
            
            return valid_detections
            
        except Exception as e:
            logger.error(f"Error detectando múltiples rostros: {str(e)}")
            return []

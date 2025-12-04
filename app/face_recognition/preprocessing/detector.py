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
    def __init__(self, min_confidence=0.85):
        self.detector = MTCNN()
        self.min_confidence = min_confidence
        self.last_detection = None  # Cachear última detección
        logger.info(f"FaceDetector inicializado con confianza mínima: {min_confidence}")
    
    def detect_face(self, image_path=None, image_array=None):
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

            # Mejorar contraste de la imagen
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced_image = cv2.merge((cl,a,b))
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2RGB)

            # Detectar rostros en imagen mejorada
            detections = self.detector.detect_faces(enhanced_image)
        
            # Si no detecta nada, intentar con la imagen original
            if not detections:
                logger.warning("No se detectaron rostros en imagen mejorada, probando original...")
                detections = self.detector.detect_faces(image)
        
            if not detections:
                logger.warning("No se detectaron rostros en la imagen")
                return None


            # Tomar la detección con mayor confianza
            best_detection = max(detections, key=lambda x: x['confidence'])

            # Usar umbral más bajo para tiempo real
            min_threshold = self.min_confidence * 0.8  # 80% del mínimo
        
            if best_detection['confidence'] < min_threshold:
                logger.warning(
                    f"Confianza de detección ({best_detection['confidence']:.2f}) "
                    f"menor que el mínimo ({min_threshold:.2f})"
                )
                return None

            detection_result = {
                'box': best_detection['box'],
                'confidence': best_detection['confidence'],
                'keypoints': best_detection['keypoints'],
                'total_faces': len(detections)
            }
            
            # Cachear última detección válida
            self.last_detection = detection_result
            
            return detection_result
        
        except Exception as e:
            logger.error(f"Error detectando rostro: {str(e)}", exc_info=True)
            return None
    
    def align_face(self, image, keypoints):
        """Alinea el rostro usando los keypoints de los ojos"""
        try:
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            # Calcular ángulo entre ojos
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calcular centro entre ojos
            eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                          (left_eye[1] + right_eye[1]) / 2)
            
            # Matriz de rotación
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            
            # Aplicar rotación
            h, w = image.shape[:2]
            aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
            
            return aligned
        except Exception as e:
            logger.warning(f"No se pudo alinear el rostro: {str(e)}")
            return image
    
    def extract_face(self, image_path=None, image_array=None, target_size=(160, 160), align=True):
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
            
            # Alinear rostro si se especifica
            if align and 'keypoints' in detection:
                image = self.align_face(image, detection['keypoints'])
            
            # Extraer coordenadas del rostro con margen
            x, y, width, height = detection['box']
            
            # Agregar margen del 20% para capturar más contexto
            margin = 0.2
            x_margin = int(width * margin)
            y_margin = int(height * margin)
            
            x = max(0, x - x_margin)
            y = max(0, y - y_margin)
            x2 = min(image.shape[1], x + width + 2 * x_margin)
            y2 = min(image.shape[0], y + height + 2 * y_margin)
            
            # Recortar rostro
            face = image[y:y2, x:x2]
            
            # Redimensionar al tamaño objetivo
            face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_CUBIC)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error extrayendo rostro: {str(e)}")
            return None
    
    def calculate_blur_score(self, image):
        """Calcula el nivel de desenfoque usando el operador Laplaciano"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def calculate_brightness(self, image):
        """Calcula el nivel de brillo promedio de la imagen"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return np.mean(hsv[:, :, 2])
    
    def validate_face_quality(self, image_path=None, image_array=None):
        # Cargar imagen
        if image_path:
            image = cv2.imread(image_path)
            if image is None:
                return {'is_valid': False, 'quality_score': 0.0, 'message': 'No se pudo cargar la imagen'}
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_array is not None:
            image = image_array
        else:
            return {'is_valid': False, 'quality_score': 0.0, 'message': 'No se proporcionó imagen'}
        
        detection = self.detect_face(image_array=image)
        
        if detection is None:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'message': 'No se detectó rostro o confianza muy baja'
            }
        
        # Calcular score de calidad basado en varios factores
        quality_score = detection['confidence']
        messages = []
        
        # Penalizar si se detectan múltiples rostros
        if detection['total_faces'] > 1:
            quality_score *= 0.8
            messages.append(f"Se detectaron {detection['total_faces']} rostros")
        
        # Verificar tamaño del rostro detectado
        box = detection['box']
        face_area = box[2] * box[3]
        
        if face_area < 10000:
            quality_score *= 0.7
            messages.append("Rostro muy pequeño")
        
        # Extraer región del rostro para análisis
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        face_region = image[y:y2, x:x2]
        
        # Verificar desenfoque
        blur_score = self.calculate_blur_score(face_region)
        if blur_score < 100:
            quality_score *= 0.85
            messages.append("Imagen desenfocada")
        
        # Verificar luminosidad
        brightness = self.calculate_brightness(face_region)
        if brightness < 50 or brightness > 200:
            quality_score *= 0.9
            messages.append("Mala iluminación")
        
        is_valid = quality_score >= self.min_confidence * 0.85
        message = "Imagen válida" if is_valid else ", ".join(messages) if messages else "Calidad insuficiente"
        
        return {
            'is_valid': is_valid,
            'quality_score': float(quality_score),
            'confidence': float(detection['confidence']),
            'total_faces': detection['total_faces'],
            'blur_score': float(blur_score),
            'brightness': float(brightness),
            'message': message
        }
    
    def detect_multiple_faces(self, image_path=None, image_array=None):
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

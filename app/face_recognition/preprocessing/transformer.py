import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FaceTransformer:
    """
    Transformador para preprocesar imágenes faciales.
    Aplica normalización, aumento de datos y transformaciones.
    """
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size
        logger.info(f"FaceTransformer inicializado con tamaño: {target_size}")
    
    def normalize(self, image):
        image = image.astype(np.float32)
        
        # Normalizar a rango [0, 1]
        image = image / 255.0
        return image
    
    def denormalize(self, image):
        if image.min() < 0:
            image = (image / 2.0) + 0.5
        
        # Convertir a rango [0, 255]
        image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def resize(self, image, size=None):
        if size is None:
            size = self.target_size
        
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    def preprocess_for_training(self, image):
        # Redimensionar
        image = self.resize(image)
        
        # Normalizar
        image = self.normalize(image)
        
        return image
    
    def preprocess_for_inference(self, image):
        # Mismo preprocesamiento que para entrenamiento
        image = self.preprocess_for_training(image)
        
        # Agregar dimensión de batch si es necesario
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        return image
    
    def apply_histogram_equalization(self, image):
        # Convertir a YUV
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        # Ecualizar el canal Y (luminancia)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        
        # Convertir de vuelta a RGB
        image_eq = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
        
        return image_eq
    
    def augment_image(self, image):
        augmented_images = [image]  # Incluir original
        
        # Flip horizontal
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        
        # Ajustes de brillo
        brightness_factors = [0.8, 1.2]
        for factor in brightness_factors:
            bright = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            augmented_images.append(bright)
        
        # Rotaciones pequeñas
        angles = [-10, 10]
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        for angle in angles:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented_images.append(rotated)
        
        return augmented_images
    
    def add_gaussian_noise(self, image, mean=0, std=10):
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def adjust_contrast(self, image, factor=1.5):
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def convert_to_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Convertir de vuelta a 3 canales para compatibilidad
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        return gray_3channel
    
    def batch_preprocess(self, images):
        processed = []
        
        for img in images:
            processed_img = self.preprocess_for_training(img)
            processed.append(processed_img)
        
        return np.array(processed)
    
    def extract_face_encoding(self, image):
        # Asegurar tamaño correcto
        if image.shape[:2] != self.target_size:
            image = self.resize(image)
        
        # Normalizar
        image = self.normalize(image)
        
        return image

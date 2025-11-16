import os
import numpy as np
import cv2
from pathlib import Path
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class FaceDatasetManager:
    """
    Gestor de datasets para entrenamiento de reconocimiento facial.
    Organiza y prepara los datos de imágenes faciales de usuarios.
    """
    
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            dataset_dir = settings.FACE_DATASET_DIR
        
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FaceDatasetManager inicializado con directorio: {self.dataset_dir}")
    
    def create_user_directory(self, user_id):
        user_dir = self.dataset_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Directorio creado para usuario {user_id}: {user_dir}")
        return user_dir
    
    def save_face_image(self, user_id, image, image_name=None):
        user_dir = self.create_user_directory(user_id)
        
        if image_name is None:
            # Generar nombre único basado en timestamp
            import time
            image_name = f"face_{int(time.time() * 1000)}.jpg"
        
        image_path = user_dir / image_name
        
        # Guardar imagen
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convertir RGB a BGR para OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), image_bgr)
        else:
            cv2.imwrite(str(image_path), image)
        
        logger.info(f"Imagen guardada: {image_path}")
        return str(image_path)
    
    def get_user_images(self, user_id):
        user_dir = self.dataset_dir / str(user_id)
        
        if not user_dir.exists():
            logger.warning(f"No existe directorio para usuario {user_id}")
            return []
        
        # Buscar todas las imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        for ext in image_extensions:
            images.extend(user_dir.glob(f"*{ext}"))
        
        return [str(img) for img in images]
    
    def load_user_images(self, user_id):
        image_paths = self.get_user_images(user_id)
        images = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        
        logger.info(f"Cargadas {len(images)} imágenes del usuario {user_id}")
        return images
    
    def count_user_images(self, user_id):
        return len(self.get_user_images(user_id))
    
    def delete_user_images(self, user_id):
        user_dir = self.dataset_dir / str(user_id)
        
        if not user_dir.exists():
            return False
        
        # Eliminar todos los archivos
        for file in user_dir.iterdir():
            if file.is_file():
                file.unlink()
        
        # Eliminar directorio
        user_dir.rmdir()
        
        logger.info(f"Imágenes del usuario {user_id} eliminadas")
        return True
    
    def get_all_users_with_images(self):
        user_ids = []
        
        for user_dir in self.dataset_dir.iterdir():
            if user_dir.is_dir() and user_dir.name.isdigit():
                # Verificar que tenga imágenes
                if self.count_user_images(int(user_dir.name)) > 0:
                    user_ids.append(int(user_dir.name))
        
        return sorted(user_ids)
    
    def prepare_training_data(self, min_images_per_user=5):
        user_ids = self.get_all_users_with_images()
        
        # Filtrar usuarios con suficientes imágenes
        valid_users = [
            uid for uid in user_ids 
            if self.count_user_images(uid) >= min_images_per_user
        ]
        
        if not valid_users:
            logger.warning("No hay usuarios con suficientes imágenes para entrenar")
            return None, None, None
        
        # Crear mapping de user_id a índice de clase
        user_id_mapping = {uid: idx for idx, uid in enumerate(valid_users)}
        
        X_train = []
        y_train = []
        
        for user_id in valid_users:
            images = self.load_user_images(user_id)
            class_idx = user_id_mapping[user_id]
            
            for img in images:
                X_train.append(img)
                y_train.append(class_idx)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        logger.info(
            f"Dataset preparado: {len(X_train)} imágenes, "
            f"{len(valid_users)} usuarios/clases"
        )
        
        return X_train, y_train, user_id_mapping
    
    def get_dataset_statistics(self):
        user_ids = self.get_all_users_with_images()
        
        total_images = 0
        images_per_user = []
        
        for uid in user_ids:
            count = self.count_user_images(uid)
            total_images += count
            images_per_user.append(count)
        
        stats = {
            'total_users': len(user_ids),
            'total_images': total_images,
            'avg_images_per_user': np.mean(images_per_user) if images_per_user else 0,
            'min_images_per_user': min(images_per_user) if images_per_user else 0,
            'max_images_per_user': max(images_per_user) if images_per_user else 0,
            'user_ids': user_ids
        }
        
        return stats
    
    def validate_dataset(self, min_users=2, min_images_per_user=5):
        stats = self.get_dataset_statistics()
        
        issues = []
        
        if stats['total_users'] < min_users:
            issues.append(
                f"Insuficientes usuarios: {stats['total_users']} "
                f"(mínimo {min_users})"
            )
        
        if stats['min_images_per_user'] < min_images_per_user:
            issues.append(
                f"Algunos usuarios tienen pocas imágenes: "
                f"mínimo {stats['min_images_per_user']} "
                f"(requerido {min_images_per_user})"
            )
        
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'statistics': stats
        }

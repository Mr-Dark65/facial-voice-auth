import os
import numpy as np
import librosa
from pathlib import Path
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class VoiceDatasetManager:
    """
    Gestor de datasets para entrenamiento de reconocimiento de voz.
    Organiza y prepara los datos de muestras de voz de usuarios.
    """
    
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            dataset_dir = settings.VOICE_DATASET_DIR
        
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VoiceDatasetManager inicializado con directorio: {self.dataset_dir}")
    
    def create_user_directory(self, user_id):
        """Crea directorio para un usuario específico."""
        user_dir = self.dataset_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Directorio creado para usuario {user_id}: {user_dir}")
        return user_dir
    
    def save_voice_sample(self, user_id, audio_data, sample_name=None, sample_rate=16000):
        """
        Guarda una muestra de voz de un usuario.
        
        Args:
            user_id: ID del usuario
            audio_data: Array numpy con datos de audio
            sample_name: Nombre del archivo (opcional)
            sample_rate: Frecuencia de muestreo
        
        Returns:
            Ruta del archivo guardado
        """
        user_dir = self.create_user_directory(user_id)
        
        if sample_name is None:
            import time
            sample_name = f"voice_{int(time.time() * 1000)}.wav"
        
        sample_path = user_dir / sample_name
        
        # Guardar audio usando librosa
        try:
            import soundfile as sf
            sf.write(str(sample_path), audio_data, sample_rate)
            logger.info(f"Muestra de voz guardada: {sample_path}")
            return str(sample_path)
        except Exception as e:
            logger.error(f"Error guardando muestra de voz: {str(e)}")
            return None
    
    def get_user_samples(self, user_id):
        """Obtiene todas las muestras de voz de un usuario."""
        user_dir = self.dataset_dir / str(user_id)
        
        if not user_dir.exists():
            logger.warning(f"No existe directorio para usuario {user_id}")
            return []
        
        # Buscar todos los archivos de audio
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        samples = []
        
        for ext in audio_extensions:
            samples.extend(user_dir.glob(f"*{ext}"))
        
        return [str(sample) for sample in samples]
    
    def load_user_samples(self, user_id, sample_rate=16000, duration=None):
        """Carga todas las muestras de voz de un usuario."""
        sample_paths = self.get_user_samples(user_id)
        samples = []
        
        for sample_path in sample_paths:
            try:
                audio, sr = librosa.load(sample_path, sr=sample_rate, duration=duration)
                samples.append(audio)
            except Exception as e:
                logger.error(f"Error cargando {sample_path}: {str(e)}")
        
        logger.info(f"Cargadas {len(samples)} muestras del usuario {user_id}")
        return samples
    
    def count_user_samples(self, user_id):
        """Cuenta las muestras de voz de un usuario."""
        return len(self.get_user_samples(user_id))
    
    def delete_user_samples(self, user_id):
        """Elimina todas las muestras de voz de un usuario."""
        user_dir = self.dataset_dir / str(user_id)
        
        if not user_dir.exists():
            return False
        
        # Eliminar todos los archivos
        for file in user_dir.iterdir():
            if file.is_file():
                file.unlink()
        
        # Eliminar directorio
        user_dir.rmdir()
        
        logger.info(f"Muestras del usuario {user_id} eliminadas")
        return True
    
    def get_all_users_with_samples(self):
        """Obtiene lista de usuarios que tienen muestras de voz."""
        user_ids = []
        
        for user_dir in self.dataset_dir.iterdir():
            if user_dir.is_dir() and user_dir.name.isdigit():
                # Verificar que tenga muestras
                if self.count_user_samples(int(user_dir.name)) > 0:
                    user_ids.append(int(user_dir.name))
        
        return sorted(user_ids)
    
    def prepare_training_data(self, min_samples_per_user=5, sample_rate=16000, max_duration=5.0):
        """
        Prepara datos para entrenamiento.
        
        Args:
            min_samples_per_user: Mínimo de muestras requeridas por usuario
            sample_rate: Frecuencia de muestreo
            max_duration: Duración máxima por muestra en segundos
        
        Returns:
            X_train, y_train, user_id_mapping
        """
        user_ids = self.get_all_users_with_samples()
        
        # Filtrar usuarios con suficientes muestras
        valid_users = [
            uid for uid in user_ids 
            if self.count_user_samples(uid) >= min_samples_per_user
        ]
        
        if not valid_users:
            logger.warning("No hay usuarios con suficientes muestras para entrenar")
            return None, None, None
        
        # Crear mapping de user_id a índice de clase
        user_id_mapping = {uid: idx for idx, uid in enumerate(valid_users)}
        
        X_train = []
        y_train = []
        
        for user_id in valid_users:
            samples = self.load_user_samples(user_id, sample_rate, max_duration)
            class_idx = user_id_mapping[user_id]
            
            for audio in samples:
                X_train.append(audio)
                y_train.append(class_idx)
        
        X_train = np.array(X_train, dtype=object)  # Object porque pueden tener diferentes longitudes
        y_train = np.array(y_train)
        
        logger.info(
            f"Dataset preparado: {len(X_train)} muestras, "
            f"{len(valid_users)} usuarios/clases"
        )
        
        return X_train, y_train, user_id_mapping
    
    def get_dataset_statistics(self):
        """Obtiene estadísticas del dataset."""
        user_ids = self.get_all_users_with_samples()
        
        total_samples = 0
        samples_per_user = []
        durations = []
        
        for uid in user_ids:
            count = self.count_user_samples(uid)
            total_samples += count
            samples_per_user.append(count)
            
            # Calcular duraciones
            for sample_path in self.get_user_samples(uid):
                try:
                    duration = librosa.get_duration(path=sample_path)
                    durations.append(duration)
                except:
                    pass
        
        stats = {
            'total_users': len(user_ids),
            'total_samples': total_samples,
            'avg_samples_per_user': np.mean(samples_per_user) if samples_per_user else 0,
            'min_samples_per_user': min(samples_per_user) if samples_per_user else 0,
            'max_samples_per_user': max(samples_per_user) if samples_per_user else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'total_duration': sum(durations) if durations else 0,
            'user_ids': user_ids
        }
        
        return stats
    
    def validate_dataset(self, min_users=2, min_samples_per_user=5):
        """Valida que el dataset sea adecuado para entrenamiento."""
        stats = self.get_dataset_statistics()
        
        issues = []
        
        if stats['total_users'] < min_users:
            issues.append(
                f"Insuficientes usuarios: {stats['total_users']} "
                f"(mínimo {min_users})"
            )
        
        if stats['min_samples_per_user'] < min_samples_per_user:
            issues.append(
                f"Algunos usuarios tienen pocas muestras: "
                f"mínimo {stats['min_samples_per_user']} "
                f"(requerido {min_samples_per_user})"
            )
        
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'statistics': stats
        }

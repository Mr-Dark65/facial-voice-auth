import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MFCCExtractor:
    """
    Extractor de características MFCC (Mel-Frequency Cepstral Coefficients).
    Extrae características acústicas de audio para reconocimiento de voz.
    """
    
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        logger.info(f"MFCCExtractor inicializado: sr={sample_rate}, n_mfcc={n_mfcc}")
    
    def load_audio(self, audio_path, duration=None):
        """Carga un archivo de audio y lo resampling a la frecuencia objetivo."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
            return audio, sr
        except Exception as e:
            logger.error(f"Error cargando audio {audio_path}: {str(e)}")
            return None, None
    
    def extract_mfcc(self, audio_path=None, audio_array=None, delta=True, delta_delta=True):
        """
        Extrae características MFCC de un audio.
        
        Args:
            audio_path: Ruta al archivo de audio
            audio_array: Array de audio ya cargado
            delta: Si incluir delta (primera derivada)
            delta_delta: Si incluir delta-delta (segunda derivada)
        
        Returns:
            Array numpy con características MFCC
        """
        try:
            # Cargar audio
            if audio_path:
                audio, sr = self.load_audio(audio_path)
                if audio is None:
                    return None
            elif audio_array is not None:
                audio = audio_array
                sr = self.sample_rate
            else:
                logger.error("Debe proporcionar audio_path o audio_array")
                return None
            
            # Extraer MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            features = [mfccs]
            
            # Agregar deltas
            if delta:
                mfcc_delta = librosa.feature.delta(mfccs)
                features.append(mfcc_delta)
            
            if delta_delta:
                mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
                features.append(mfcc_delta2)
            
            # Concatenar características
            features = np.concatenate(features, axis=0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo MFCC: {str(e)}")
            return None
    
    def extract_spectral_features(self, audio_path=None, audio_array=None):
        """Extrae características espectrales adicionales."""
        try:
            if audio_path:
                audio, sr = self.load_audio(audio_path)
                if audio is None:
                    return None
            elif audio_array is not None:
                audio = audio_array
                sr = self.sample_rate
            else:
                return None
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio, frame_length=self.n_fft, hop_length=self.hop_length
            )[0]
            
            return {
                'spectral_centroid': spectral_centroids,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zcr
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo características espectrales: {str(e)}")
            return None
    
    def normalize_features(self, features):
        """Normaliza características usando mean y std."""
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        normalized = (features - mean) / (std + 1e-8)
        return normalized
    
    def pad_or_truncate(self, features, target_length):
        """Ajusta la longitud de las características al tamaño objetivo."""
        current_length = features.shape[1]
        
        if current_length < target_length:
            # Padding con zeros
            pad_width = target_length - current_length
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        elif current_length > target_length:
            # Truncar
            features = features[:, :target_length]
        
        return features
    
    def extract_statistical_features(self, mfccs):
        """Extrae estadísticas de los MFCCs a lo largo del tiempo."""
        features = []
        
        # Media
        features.append(np.mean(mfccs, axis=1))
        
        # Desviación estándar
        features.append(np.std(mfccs, axis=1))
        
        # Máximo
        features.append(np.max(mfccs, axis=1))
        
        # Mínimo
        features.append(np.min(mfccs, axis=1))
        
        # Mediana
        features.append(np.median(mfccs, axis=1))
        
        return np.concatenate(features)
    
    def get_audio_duration(self, audio_path=None, audio_array=None):
        """Obtiene la duración de un audio en segundos."""
        try:
            if audio_path:
                duration = librosa.get_duration(path=audio_path)
            elif audio_array is not None:
                duration = len(audio_array) / self.sample_rate
            else:
                return None
            
            return duration
            
        except Exception as e:
            logger.error(f"Error obteniendo duración: {str(e)}")
            return None

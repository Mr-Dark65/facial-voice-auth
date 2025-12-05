import librosa
import numpy as np
import noisereduce as nr
import logging

logger = logging.getLogger(__name__)


class AudioCleaner:
    """
    Limpiador de audio para preprocesamiento.
    Aplica reducción de ruido, normalización y filtrado.
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        logger.info(f"AudioCleaner inicializado con sample_rate={sample_rate}")
    
    def load_audio(self, audio_path, duration=None):
        """Carga un archivo de audio."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
            return audio, sr
        except Exception as e:
            logger.error(f"Error cargando audio: {str(e)}")
            return None, None
    
    def reduce_noise(self, audio_array, noise_profile=None):
        """
        Reduce el ruido de fondo del audio.
        
        Args:
            audio_array: Array de audio
            noise_profile: Perfil de ruido (si None, se estima del audio)
        
        Returns:
            Audio con ruido reducido
        """
        try:
            # Usar noisereduce para reducción de ruido
            cleaned_audio = nr.reduce_noise(
                y=audio_array,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=0.8
            )
            
            return cleaned_audio
            
        except Exception as e:
            logger.error(f"Error reduciendo ruido: {str(e)}")
            return audio_array
    
    def normalize_audio(self, audio_array, target_level=-20.0):
        """
        Normaliza el volumen del audio.
        
        Args:
            audio_array: Array de audio
            target_level: Nivel objetivo en dB
        
        Returns:
            Audio normalizado
        """
        try:
            # Calcular RMS actual
            rms = np.sqrt(np.mean(audio_array**2))
            
            if rms == 0:
                return audio_array
            
            # Calcular factor de escala
            current_level = 20 * np.log10(rms)
            scale = 10 ** ((target_level - current_level) / 20)
            
            # Aplicar normalización
            normalized = audio_array * scale
            
            # Clip para evitar valores fuera de rango
            normalized = np.clip(normalized, -1.0, 1.0)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizando audio: {str(e)}")
            return audio_array
    
    def trim_silence(self, audio_array, top_db=20):
        """
        Elimina silencios al inicio y final del audio.
        
        Args:
            audio_array: Array de audio
            top_db: Umbral de silencio en dB
        
        Returns:
            Audio sin silencios
        """
        try:
            trimmed, _ = librosa.effects.trim(audio_array, top_db=top_db)
            return trimmed
            
        except Exception as e:
            logger.error(f"Error eliminando silencios: {str(e)}")
            return audio_array
    
    def apply_preemphasis(self, audio_array, coef=0.97):
        """
        Aplica pre-énfasis para realzar altas frecuencias.
        
        Args:
            audio_array: Array de audio
            coef: Coeficiente de pre-énfasis
        
        Returns:
            Audio con pre-énfasis aplicado
        """
        try:
            emphasized = np.append(audio_array[0], audio_array[1:] - coef * audio_array[:-1])
            return emphasized
            
        except Exception as e:
            logger.error(f"Error aplicando pre-énfasis: {str(e)}")
            return audio_array
    
    def calculate_snr(self, audio_array):
        """
        Calcula la relación señal-ruido (SNR) del audio.
        
        Returns:
            SNR en dB
        """
        try:
            # Estimar señal (partes con mayor energía)
            frame_length = int(0.02 * self.sample_rate)  # 20ms frames
            frames = librosa.util.frame(audio_array, frame_length=frame_length, hop_length=frame_length)
            frame_energy = np.sum(frames**2, axis=0)
            
            # Top 50% como señal, bottom 25% como ruido
            sorted_energy = np.sort(frame_energy)
            signal_threshold = sorted_energy[int(len(sorted_energy) * 0.5)]
            noise_threshold = sorted_energy[int(len(sorted_energy) * 0.25)]
            
            signal_power = np.mean(frame_energy[frame_energy >= signal_threshold])
            noise_power = np.mean(frame_energy[frame_energy <= noise_threshold])
            
            if noise_power == 0:
                return float('inf')
            
            snr = 10 * np.log10(signal_power / noise_power)
            
            return snr
            
        except Exception as e:
            logger.error(f"Error calculando SNR: {str(e)}")
            return None
    
    def validate_audio_quality(self, audio_path=None, audio_array=None, min_duration=1.0, max_duration=30.0, min_snr=10.0):
        """
        Valida la calidad del audio.
        
        Returns:
            dict con información de validación
        """
        try:
            if audio_path:
                audio, sr = self.load_audio(audio_path)
                if audio is None:
                    return {'is_valid': False, 'message': 'No se pudo cargar el audio'}
            elif audio_array is not None:
                audio = audio_array
                sr = self.sample_rate
            else:
                return {'is_valid': False, 'message': 'No se proporcionó audio'}
            
            # Duración
            duration = len(audio) / sr
            
            if duration < min_duration:
                return {
                    'is_valid': False,
                    'message': f'Audio muy corto ({duration:.1f}s < {min_duration}s)',
                    'duration': duration
                }
            
            if duration > max_duration:
                return {
                    'is_valid': False,
                    'message': f'Audio muy largo ({duration:.1f}s > {max_duration}s)',
                    'duration': duration
                }
            
            # SNR
            snr = self.calculate_snr(audio)
            
            if snr is None:
                return {
                    'is_valid': False,
                    'message': 'No se pudo calcular SNR',
                    'duration': duration
                }
            
            if snr < min_snr:
                return {
                    'is_valid': False,
                    'message': f'SNR muy bajo ({snr:.1f}dB < {min_snr}dB). Demasiado ruido.',
                    'duration': duration,
                    'snr': snr
                }
            
            # Calcular score de calidad
            quality_score = min(1.0, (snr / 30.0))  # Normalizado a [0, 1]
            
            return {
                'is_valid': True,
                'message': 'Audio válido',
                'duration': duration,
                'snr': snr,
                'quality_score': quality_score
            }
            
        except Exception as e:
            logger.error(f"Error validando calidad: {str(e)}")
            return {'is_valid': False, 'message': f'Error: {str(e)}'}
    
    def preprocess_audio(self, audio_path=None, audio_array=None, reduce_noise_flag=True, normalize_flag=True, trim_flag=True):
        """
        Pipeline completo de preprocesamiento de audio.
        
        Returns:
            Audio preprocesado
        """
        try:
            if audio_path:
                audio, sr = self.load_audio(audio_path)
                if audio is None:
                    return None
            elif audio_array is not None:
                audio = audio_array
            else:
                return None
            
            # Trim silencio
            if trim_flag:
                audio = self.trim_silence(audio)
            
            # Reducir ruido
            if reduce_noise_flag:
                audio = self.reduce_noise(audio)
            
            # Normalizar
            if normalize_flag:
                audio = self.normalize_audio(audio)
            
            # Pre-énfasis
            audio = self.apply_preemphasis(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            return None

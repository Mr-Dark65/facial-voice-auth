from django.db import models
from django.conf import settings
import os


class VoiceSample(models.Model):
    """
    Modelo para almacenar muestras de voz de usuarios.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='voice_samples',
        verbose_name='Usuario'
    )
    audio_file = models.FileField(
        upload_to='voice_samples/',
        verbose_name='Archivo de audio'
    )
    embedding = models.JSONField(
        null=True,
        blank=True,
        verbose_name='Embedding de voz',
        help_text='Vector de características extraído del audio'
    )
    
    # Metadata del audio
    duration = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Duración (segundos)'
    )
    sample_rate = models.IntegerField(
        null=True,
        blank=True,
        verbose_name='Frecuencia de muestreo (Hz)'
    )
    
    # Calidad
    is_valid = models.BooleanField(
        default=True,
        verbose_name='Es válida',
        help_text='Indica si la muestra es válida para entrenamiento'
    )
    quality_score = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Score de calidad',
        help_text='Puntuación de calidad del audio'
    )
    snr = models.FloatField(
        null=True,
        blank=True,
        verbose_name='SNR (Signal-to-Noise Ratio)',
        help_text='Relación señal-ruido en dB'
    )
    
    # Timestamps
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de subida'
    )
    processed_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name='Fecha de procesamiento'
    )
    
    class Meta:
        verbose_name = 'Muestra de voz'
        verbose_name_plural = 'Muestras de voz'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"Voice sample {self.id} - User: {self.user.username}"
    
    def delete(self, *args, **kwargs):
        """Elimina el archivo físico al borrar el registro."""
        if self.audio_file and os.path.isfile(self.audio_file.path):
            os.remove(self.audio_file.path)
        super().delete(*args, **kwargs)


class VoiceRecognitionModel(models.Model):
    """
    Modelo para almacenar información sobre modelos de voz entrenados.
    """
    name = models.CharField(
        max_length=100,
        unique=True,
        verbose_name='Nombre del modelo'
    )
    version = models.CharField(
        max_length=20,
        verbose_name='Versión'
    )
    model_file = models.CharField(
        max_length=255,
        verbose_name='Archivo del modelo',
        help_text='Ruta al archivo .h5 o .keras del modelo'
    )
    
    # Métricas de entrenamiento
    accuracy = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Precisión'
    )
    loss = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Pérdida'
    )
    val_accuracy = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Precisión de validación'
    )
    val_loss = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Pérdida de validación'
    )
    
    # Información de entrenamiento
    training_samples = models.IntegerField(
        default=0,
        verbose_name='Muestras de entrenamiento'
    )
    epochs_trained = models.IntegerField(
        default=0,
        verbose_name='Épocas entrenadas'
    )
    architecture = models.TextField(
        blank=True,
        verbose_name='Arquitectura',
        help_text='Descripción de la arquitectura del modelo'
    )
    
    # Estado
    is_active = models.BooleanField(
        default=False,
        verbose_name='Modelo activo',
        help_text='Indica si este es el modelo en uso actualmente'
    )
    
    # Timestamps
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Fecha de creación'
    )
    trained_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name='Fecha de entrenamiento'
    )
    
    class Meta:
        verbose_name = 'Modelo de reconocimiento de voz'
        verbose_name_plural = 'Modelos de reconocimiento de voz'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} v{self.version} {'(Activo)' if self.is_active else ''}"
    
    def save(self, *args, **kwargs):
        """Si se marca como activo, desactiva los demás modelos."""
        if self.is_active:
            VoiceRecognitionModel.objects.filter(is_active=True).update(is_active=False)
        super().save(*args, **kwargs)

from django.db import models
from django.conf import settings
import os


class FaceImage(models.Model):
    """
    Modelo para almacenar imágenes faciales de usuarios.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='face_images',
        verbose_name='Usuario'
    )
    image = models.ImageField(
        upload_to='face_images/',
        verbose_name='Imagen facial'
    )
    embedding = models.JSONField(
        null=True,
        blank=True,
        verbose_name='Embedding facial',
        help_text='Vector de características extraído de la imagen'
    )
    
    # Metadata
    is_valid = models.BooleanField(
        default=True,
        verbose_name='Es válida',
        help_text='Indica si la imagen es válida para entrenamiento'
    )
    quality_score = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Score de calidad',
        help_text='Puntuación de calidad de la imagen detectada'
    )
    face_detected = models.BooleanField(
        default=False,
        verbose_name='Rostro detectado',
        help_text='Indica si se detectó un rostro en la imagen'
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
        verbose_name = 'Imagen facial'
        verbose_name_plural = 'Imágenes faciales'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"Face image {self.id} - User: {self.user.username}"
    
    def delete(self, *args, **kwargs):
        """Elimina el archivo físico al borrar el registro."""
        if self.image and os.path.isfile(self.image.path):
            os.remove(self.image.path)
        super().delete(*args, **kwargs)


class FaceRecognitionModel(models.Model):
    """
    Modelo para almacenar información sobre modelos entrenados.
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
        help_text='Ruta al archivo .h5 del modelo'
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
        verbose_name = 'Modelo de reconocimiento facial'
        verbose_name_plural = 'Modelos de reconocimiento facial'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} v{self.version} {'(Activo)' if self.is_active else ''}"
    
    def save(self, *args, **kwargs):
        """Si se marca como activo, desactiva los demás modelos."""
        if self.is_active:
            FaceRecognitionModel.objects.filter(is_active=True).update(is_active=False)
        super().save(*args, **kwargs)

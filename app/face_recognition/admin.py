from django.contrib import admin
from .models import FaceImage, FaceRecognitionModel


@admin.register(FaceImage)
class FaceImageAdmin(admin.ModelAdmin):
    """
    Administración de imágenes faciales.
    """
    list_display = [
        'id', 'user', 'is_valid', 'face_detected',
        'quality_score', 'uploaded_at', 'processed_at'
    ]
    list_filter = ['is_valid', 'face_detected', 'uploaded_at']
    search_fields = ['user__email', 'user__username']
    readonly_fields = [
        'uploaded_at', 'processed_at', 'quality_score',
        'face_detected', 'embedding'
    ]
    ordering = ['-uploaded_at']
    
    fieldsets = (
        ('Usuario', {
            'fields': ('user', 'image')
        }),
        ('Validación', {
            'fields': ('is_valid', 'face_detected', 'quality_score')
        }),
        ('Embedding', {
            'fields': ('embedding',),
            'classes': ('collapse',)
        }),
        ('Fechas', {
            'fields': ('uploaded_at', 'processed_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(FaceRecognitionModel)
class FaceRecognitionModelAdmin(admin.ModelAdmin):
    """
    Administración de modelos de reconocimiento facial.
    """
    list_display = [
        'name', 'version', 'is_active',
        'accuracy', 'val_accuracy',
        'training_samples', 'epochs_trained',
        'created_at', 'trained_at'
    ]
    list_filter = ['is_active', 'created_at', 'trained_at']
    search_fields = ['name', 'version']
    readonly_fields = [
        'created_at', 'trained_at', 'model_file',
        'accuracy', 'loss', 'val_accuracy', 'val_loss',
        'training_samples', 'epochs_trained'
    ]
    ordering = ['-created_at']
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('name', 'version', 'is_active')
        }),
        ('Archivo', {
            'fields': ('model_file',)
        }),
        ('Métricas de Entrenamiento', {
            'fields': (
                'accuracy', 'loss', 'val_accuracy', 'val_loss',
                'training_samples', 'epochs_trained'
            )
        }),
        ('Arquitectura', {
            'fields': ('architecture',),
            'classes': ('collapse',)
        }),
        ('Fechas', {
            'fields': ('created_at', 'trained_at')
        }),
    )
    
    def has_delete_permission(self, request, obj=None):
        """No permitir eliminar el modelo activo."""
        if obj and obj.is_active:
            return False
        return super().has_delete_permission(request, obj)

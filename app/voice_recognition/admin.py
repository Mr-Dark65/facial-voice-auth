from django.contrib import admin
from .models import VoiceSample, VoiceRecognitionModel


@admin.register(VoiceSample)
class VoiceSampleAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'duration', 'quality_score', 'snr', 'is_valid', 'uploaded_at']
    list_filter = ['is_valid', 'uploaded_at', 'processed_at']
    search_fields = ['user__username', 'user__email']
    readonly_fields = ['uploaded_at', 'processed_at']
    
    fieldsets = (
        ('Usuario', {
            'fields': ('user',)
        }),
        ('Audio', {
            'fields': ('audio_file', 'duration', 'sample_rate')
        }),
        ('Calidad', {
            'fields': ('is_valid', 'quality_score', 'snr')
        }),
        ('Embedding', {
            'fields': ('embedding',),
            'classes': ('collapse',)
        }),
        ('Fechas', {
            'fields': ('uploaded_at', 'processed_at')
        }),
    )


@admin.register(VoiceRecognitionModel)
class VoiceRecognitionModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'version', 'is_active', 'accuracy', 'val_accuracy', 'training_samples', 'trained_at']
    list_filter = ['is_active', 'trained_at']
    search_fields = ['name', 'version']
    readonly_fields = ['created_at', 'trained_at']
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('name', 'version', 'model_file', 'is_active')
        }),
        ('Métricas', {
            'fields': ('accuracy', 'loss', 'val_accuracy', 'val_loss')
        }),
        ('Entrenamiento', {
            'fields': ('training_samples', 'epochs_trained', 'architecture')
        }),
        ('Fechas', {
            'fields': ('created_at', 'trained_at')
        }),
    )

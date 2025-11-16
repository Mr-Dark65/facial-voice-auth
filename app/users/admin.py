from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, AuthenticationAttempt


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """
    Administración personalizada para el modelo User.
    """
    list_display = [
        'email', 'username', 'first_name', 'last_name',
        'face_registered', 'voice_registered', 'multimodal_registered',
        'is_active', 'is_staff', 'date_joined'
    ]
    list_filter = [
        'is_active', 'is_staff', 'is_superuser',
        'face_registered', 'voice_registered', 'multimodal_registered',
        'date_joined'
    ]
    search_fields = ['email', 'username', 'first_name', 'last_name']
    ordering = ['-date_joined']
    
    fieldsets = (
        ('Información básica', {
            'fields': ('email', 'username', 'password')
        }),
        ('Información personal', {
            'fields': ('first_name', 'last_name')
        }),
        ('Datos biométricos - Rostro', {
            'fields': ('face_registered', 'face_embedding', 'face_images_count'),
            'classes': ('collapse',)
        }),
        ('Datos biométricos - Voz', {
            'fields': ('voice_registered', 'voice_embedding', 'voice_samples_count'),
            'classes': ('collapse',)
        }),
        ('Estado multimodal', {
            'fields': ('multimodal_registered',)
        }),
        ('Permisos', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions'),
            'classes': ('collapse',)
        }),
        ('Fechas importantes', {
            'fields': ('last_login', 'date_joined'),
            'classes': ('collapse',)
        }),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'username', 'password1', 'password2', 'is_staff', 'is_active')
        }),
    )
    
    readonly_fields = ['date_joined', 'last_login']


@admin.register(AuthenticationAttempt)
class AuthenticationAttemptAdmin(admin.ModelAdmin):
    """
    Administración para intentos de autenticación.
    """
    list_display = [
        'email_attempted', 'user', 'status',
        'face_confidence', 'voice_confidence', 'combined_confidence',
        'ip_address', 'timestamp'
    ]
    list_filter = ['status', 'timestamp']
    search_fields = ['email_attempted', 'user__email', 'user__username', 'ip_address']
    readonly_fields = [
        'user', 'email_attempted', 'status',
        'face_confidence', 'voice_confidence', 'combined_confidence',
        'ip_address', 'user_agent', 'timestamp'
    ]
    ordering = ['-timestamp']
    
    def has_add_permission(self, request):
        """No permitir agregar intentos manualmente."""
        return False
    
    def has_change_permission(self, request, obj=None):
        """No permitir modificar intentos."""
        return False


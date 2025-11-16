from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.utils import timezone
import os


class UserManager(BaseUserManager):
    """
    Gestor personalizado para el modelo de usuario.
    """
    def create_user(self, email, username, password=None, **extra_fields):
        """
        Crea y guarda un usuario regular.
        """
        if not email:
            raise ValueError('El usuario debe tener un email')
        if not username:
            raise ValueError('El usuario debe tener un username')
        
        email = self.normalize_email(email)
        user = self.model(email=email, username=username, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, username, password=None, **extra_fields):
        """
        Crea y guarda un superusuario.
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser debe tener is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser debe tener is_superuser=True.')
        
        return self.create_user(email, username, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    """
    Modelo de usuario personalizado con soporte para autenticación multimodal.
    Incluye campos para almacenar datos biométricos de rostro y voz.
    """
    email = models.EmailField(unique=True, verbose_name='Email')
    username = models.CharField(max_length=150, unique=True, verbose_name='Nombre de usuario')
    first_name = models.CharField(max_length=100, blank=True, verbose_name='Nombre')
    last_name = models.CharField(max_length=100, blank=True, verbose_name='Apellido')
    
    # Campos de estado
    is_active = models.BooleanField(default=True, verbose_name='Activo')
    is_staff = models.BooleanField(default=False, verbose_name='Staff')
    is_superuser = models.BooleanField(default=False, verbose_name='Superusuario')
    
    # Fechas importantes
    date_joined = models.DateTimeField(default=timezone.now, verbose_name='Fecha de registro')
    last_login = models.DateTimeField(null=True, blank=True, verbose_name='Último login')
    
    # Campos para autenticación biométrica - Rostro
    face_registered = models.BooleanField(default=False, verbose_name='Rostro registrado')
    face_embedding = models.JSONField(
        null=True, 
        blank=True, 
        verbose_name='Embedding facial',
        help_text='Vector de características del rostro'
    )
    face_images_count = models.IntegerField(default=0, verbose_name='Número de imágenes faciales')
    
    # Campos para autenticación biométrica - Voz
    voice_registered = models.BooleanField(default=False, verbose_name='Voz registrada')
    voice_embedding = models.JSONField(
        null=True, 
        blank=True, 
        verbose_name='Embedding de voz',
        help_text='Vector de características de la voz'
    )
    voice_samples_count = models.IntegerField(default=0, verbose_name='Número de muestras de voz')
    
    # Autenticación multimodal completada
    multimodal_registered = models.BooleanField(
        default=False, 
        verbose_name='Autenticación multimodal completa'
    )
    
    # Configuración del gestor
    objects = UserManager()
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    class Meta:
        verbose_name = 'Usuario'
        verbose_name_plural = 'Usuarios'
        ordering = ['-date_joined']
    
    def __str__(self):
        return f"{self.username} ({self.email})"
    
    def get_full_name(self):
        """Retorna el nombre completo del usuario."""
        return f"{self.first_name} {self.last_name}".strip() or self.username
    
    def get_short_name(self):
        """Retorna el nombre corto del usuario."""
        return self.first_name or self.username
    
    @property
    def is_biometric_complete(self):
        """Verifica si el usuario tiene ambos datos biométricos registrados."""
        return self.face_registered and self.voice_registered
    
    def get_face_directory(self):
        """Retorna el directorio donde se guardan las imágenes faciales del usuario."""
        return os.path.join('media', 'face_images', str(self.id))
    
    def get_voice_directory(self):
        """Retorna el directorio donde se guardan las muestras de voz del usuario."""
        return os.path.join('media', 'voice_samples', str(self.id))


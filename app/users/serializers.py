from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import User, AuthenticationAttempt


class UserRegistrationSerializer(serializers.ModelSerializer):
    """
    Serializer para registro de nuevos usuarios.
    """
    password = serializers.CharField(
        write_only=True, 
        required=True, 
        style={'input_type': 'password'},
        min_length=8
    )
    password_confirm = serializers.CharField(
        write_only=True, 
        required=True, 
        style={'input_type': 'password'}
    )
    
    class Meta:
        model = User
        fields = [
            'id', 'email', 'username', 'first_name', 'last_name',
            'password', 'password_confirm'
        ]
        read_only_fields = ['id']
    
    def validate(self, attrs):
        """Validación de contraseñas coincidentes."""
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({
                "password": "Las contraseñas no coinciden."
            })
        return attrs
    
    def validate_email(self, value):
        """Validación de email único."""
        if User.objects.filter(email=value.lower()).exists():
            raise serializers.ValidationError("Este email ya está registrado.")
        return value.lower()
    
    def validate_username(self, value):
        """Validación de username único."""
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("Este nombre de usuario ya está en uso.")
        return value
    
    def create(self, validated_data):
        """Crea un nuevo usuario."""
        validated_data.pop('password_confirm')
        password = validated_data.pop('password')
        
        user = User.objects.create_user(
            password=password,
            **validated_data
        )
        return user


class UserSerializer(serializers.ModelSerializer):
    """
    Serializer para mostrar información de usuario.
    """
    full_name = serializers.CharField(source='get_full_name', read_only=True)
    is_biometric_complete = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = User
        fields = [
            'id', 'email', 'username', 'first_name', 'last_name',
            'full_name', 'date_joined', 'last_login',
            'face_registered', 'voice_registered', 'multimodal_registered',
            'is_biometric_complete', 'face_images_count', 'voice_samples_count'
        ]
        read_only_fields = [
            'id', 'date_joined', 'last_login', 'face_registered',
            'voice_registered', 'multimodal_registered', 'face_images_count',
            'voice_samples_count'
        ]


class UserUpdateSerializer(serializers.ModelSerializer):
    """
    Serializer para actualizar información básica del usuario.
    """
    class Meta:
        model = User
        fields = ['first_name', 'last_name']


class ChangePasswordSerializer(serializers.Serializer):
    """
    Serializer para cambio de contraseña.
    """
    old_password = serializers.CharField(required=True, style={'input_type': 'password'})
    new_password = serializers.CharField(
        required=True, 
        style={'input_type': 'password'},
        min_length=8
    )
    new_password_confirm = serializers.CharField(
        required=True, 
        style={'input_type': 'password'}
    )
    
    def validate(self, attrs):
        """Validación de contraseñas nuevas coincidentes."""
        if attrs['new_password'] != attrs['new_password_confirm']:
            raise serializers.ValidationError({
                "new_password": "Las contraseñas nuevas no coinciden."
            })
        return attrs
    
    def validate_old_password(self, value):
        """Valida que la contraseña antigua sea correcta."""
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError("La contraseña actual es incorrecta.")
        return value


class LoginSerializer(serializers.Serializer):
    """
    Serializer para login tradicional (antes de autenticación biométrica).
    """
    email = serializers.EmailField(required=True)
    password = serializers.CharField(
        required=True, 
        style={'input_type': 'password'},
        write_only=True
    )
    
    def validate(self, attrs):
        """Valida las credenciales del usuario."""
        email = attrs.get('email', '').lower()
        password = attrs.get('password')
        
        if email and password:
            # Intentar autenticar al usuario
            user = authenticate(
                request=self.context.get('request'),
                username=email,
                password=password
            )
            
            if not user:
                raise serializers.ValidationError(
                    "No se pudo iniciar sesión con estas credenciales."
                )
            
            if not user.is_active:
                raise serializers.ValidationError(
                    "Esta cuenta ha sido desactivada."
                )
            
            attrs['user'] = user
            return attrs
        else:
            raise serializers.ValidationError(
                "Debe incluir email y contraseña."
            )


class BiometricStatusSerializer(serializers.ModelSerializer):
    """
    Serializer para mostrar el estado de registro biométrico.
    """
    is_biometric_complete = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = User
        fields = [
            'face_registered', 'voice_registered', 
            'multimodal_registered', 'is_biometric_complete',
            'face_images_count', 'voice_samples_count'
        ]
        read_only_fields = fields


class AuthenticationAttemptSerializer(serializers.ModelSerializer):
    """
    Serializer para registros de intentos de autenticación.
    """
    user_email = serializers.EmailField(source='user.email', read_only=True)
    
    class Meta:
        model = AuthenticationAttempt
        fields = [
            'id', 'user', 'user_email', 'email_attempted', 'status',
            'face_confidence', 'voice_confidence', 'combined_confidence',
            'ip_address', 'timestamp'
        ]
        read_only_fields = fields

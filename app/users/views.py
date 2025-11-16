from django.shortcuts import render
from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token
from django.contrib.auth import login, logout
from django.utils import timezone
from .models import User, AuthenticationAttempt
from .serializers import (
    UserRegistrationSerializer, UserSerializer, UserUpdateSerializer,
    ChangePasswordSerializer, LoginSerializer, BiometricStatusSerializer,
    AuthenticationAttemptSerializer
)


class UserRegistrationView(generics.CreateAPIView):
    """
    Vista para registro de nuevos usuarios.
    POST /api/users/register/
    """
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny]
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Crear token de autenticación
        token, created = Token.objects.get_or_create(user=user)
        
        return Response({
            'message': 'Usuario creado exitosamente',
            'user': UserSerializer(user).data,
            'token': token.key
        }, status=status.HTTP_201_CREATED)


class LoginView(APIView):
    """
    Vista para login tradicional (con email y password).
    POST /api/users/login/
    """
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = LoginSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        
        user = serializer.validated_data['user']
        
        # Actualizar último login
        user.last_login = timezone.now()
        user.save(update_fields=['last_login'])
        
        # Crear o recuperar token
        token, created = Token.objects.get_or_create(user=user)
        
        return Response({
            'message': 'Login exitoso',
            'user': UserSerializer(user).data,
            'token': token.key,
            'biometric_status': BiometricStatusSerializer(user).data
        }, status=status.HTTP_200_OK)


class LogoutView(APIView):
    """
    Vista para cerrar sesión.
    POST /api/users/logout/
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        # Eliminar el token del usuario
        request.user.auth_token.delete()
        logout(request)
        
        return Response({
            'message': 'Sesión cerrada exitosamente'
        }, status=status.HTTP_200_OK)


class UserProfileView(generics.RetrieveUpdateAPIView):
    """
    Vista para ver y actualizar el perfil del usuario autenticado.
    GET/PUT /api/users/profile/
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'PUT' or self.request.method == 'PATCH':
            return UserUpdateSerializer
        return UserSerializer
    
    def get_object(self):
        return self.request.user
    
    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        
        return Response({
            'message': 'Perfil actualizado exitosamente',
            'user': UserSerializer(instance).data
        })


class ChangePasswordView(APIView):
    """
    Vista para cambiar contraseña.
    POST /api/users/change-password/
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        serializer = ChangePasswordSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        
        user = request.user
        user.set_password(serializer.validated_data['new_password'])
        user.save()
        
        # Regenerar token
        Token.objects.filter(user=user).delete()
        token = Token.objects.create(user=user)
        
        return Response({
            'message': 'Contraseña cambiada exitosamente',
            'token': token.key
        }, status=status.HTTP_200_OK)


class BiometricStatusView(APIView):
    """
    Vista para consultar el estado de registro biométrico del usuario.
    GET /api/users/biometric-status/
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        serializer = BiometricStatusSerializer(request.user)
        return Response(serializer.data)


class UserListView(generics.ListAPIView):
    """
    Vista para listar usuarios (solo admin).
    GET /api/users/list/
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]


class AuthenticationAttemptListView(generics.ListAPIView):
    """
    Vista para ver historial de intentos de autenticación del usuario.
    GET /api/users/auth-attempts/
    """
    serializer_class = AuthenticationAttemptSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return AuthenticationAttempt.objects.filter(user=self.request.user)



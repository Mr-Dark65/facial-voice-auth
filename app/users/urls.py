from django.urls import path
from .views import (
    UserRegistrationView, LoginView, LogoutView, UserProfileView,
    ChangePasswordView, BiometricStatusView, UserListView,
    AuthenticationAttemptListView, UserDeleteView
)

app_name = 'users'

urlpatterns = [
    # Autenticación
    path('register/', UserRegistrationView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    
    # Perfil de usuario
    path('profile/', UserProfileView.as_view(), name='profile'),
    path('change-password/', ChangePasswordView.as_view(), name='change-password'),
    path('delete/', UserDeleteView.as_view(), name='delete'),
    
    # Estado biométrico
    path('biometric-status/', BiometricStatusView.as_view(), name='biometric-status'),
    
    # Historial
    path('auth-attempts/', AuthenticationAttemptListView.as_view(), name='auth-attempts'),
    
    # Admin
    path('list/', UserListView.as_view(), name='user-list'),
]

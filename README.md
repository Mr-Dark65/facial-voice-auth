# Sistema de Autenticación Multimodal

Sistema de autenticación biométrica usando reconocimiento facial y de voz con Django y modelos propios (no preentrenados).

## 📋 Estructura del Proyecto

```
multimodal_auth/
├── app/
│   ├── users/              
│   ├── face_recognition/   
│   ├── voice_recognition/  
│   └── multimodal/         
├── dataset/
│   ├── faces/
│   └── voices/
├── media/
│   ├── face_images/
│   ├── voice_samples/
│   └── auth_attempts/
├── models_storage/
├── static/
├── templates/
└── multimodal_auth/
```


### Características Implementadas

1. **Modelo de Usuario Personalizado**
   - Autenticación por email
   - Campos para datos biométricos (rostro y voz)
   - Tracking de estado de registro biométrico
   - Embeddings JSON para almacenar características

2. **Modelo de Intentos de Autenticación**
   - Registro de todos los intentos de login
   - Scores de confianza para cada modalidad
   - Metadata (IP, user agent, timestamp)

3. **API REST Completa**
   - Registro de usuarios
   - Login/Logout
   - Gestión de perfil
   - Cambio de contraseña
   - Estado biométrico
   - Historial de autenticación
   - Eliminación de cuenta

### Endpoints Disponibles

```
POST   /api/users/register/           - Registrar nuevo usuario
POST   /api/users/login/              - Login con email/password
POST   /api/users/logout/             - Cerrar sesión
GET    /api/users/profile/            - Ver perfil
PUT    /api/users/profile/            - Actualizar perfil
POST   /api/users/change-password/    - Cambiar contraseña
GET    /api/users/biometric-status/   - Estado de registro biométrico
GET    /api/users/auth-attempts/      - Historial de intentos
DELETE /api/users/delete/             - Eliminar cuenta
GET    /api/users/list/               - Listar usuarios (admin)
```

## Configuración Inicial

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

Crear archivo `.env` en la raíz:

```env
DB_NAME=multimodal_auth
DB_USER=postgres
DB_PASSWORD=tu_password
DB_HOST=localhost
DB_PORT=5432
```

### 3. Aplicar Migraciones

```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. Crear Superusuario

```bash
python manage.py createsuperuser
```

### 5. Ejecutar Servidor

```bash
python manage.py runserver
```

## Uso del Módulo de Usuarios

### Ejemplo: Registrar Usuario

```python
import requests

url = "http://localhost:8000/api/users/register/"
data = {
    "email": "usuario@example.com",
    "username": "usuario123",
    "first_name": "Juan",
    "last_name": "Pérez",
    "password": "password123",
    "password_confirm": "password123"
}

response = requests.post(url, json=data)
print(response.json())
```

### Ejemplo: Login

```python
url = "http://localhost:8000/api/users/login/"
data = {
    "email": "usuario@example.com",
    "password": "password123"
}

response = requests.post(url, json=data)
token = response.json()['token']
print(f"Token: {token}")
```

### Ejemplo: Ver Perfil (Autenticado)

```python
url = "http://localhost:8000/api/users/profile/"
headers = {"Authorization": f"Token {token}"}

response = requests.get(url, headers=headers)
print(response.json())
```

## Próximos Pasos

1. **Módulo de Reconocimiento Facial** (`face_recognition`)
   - Preprocesamiento de imágenes
   - Entrenamiento de modelo CNN
   - Sistema de inferencia

2. **Módulo de Reconocimiento de Voz** (`voice_recognition`)
   - Extracción de características MFCC
   - Entrenamiento de modelo de voz
   - Sistema de inferencia

3. **Módulo Multimodal** (`multimodal`)
   - Fusión de características
   - Sistema de decisión combinado
   - Interfaz web completa

## Dependencias Principales

- Django 5.2.8
- Django REST Framework 3.15.2
- TensorFlow 2.18.0
- Keras 3.8.0
- OpenCV 4.10.0
- Librosa 0.10.2

## Configuraciones de Seguridad

El proyecto incluye configuraciones para:
- Umbrales de confianza ajustables
- Mínimo/máximo de muestras biométricas
- CORS configurado
- Autenticación por token

## Licencia

Este es un proyecto educativo para demostración de autenticación multimodal.

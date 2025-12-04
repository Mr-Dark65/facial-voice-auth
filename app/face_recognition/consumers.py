# app/face_recognition/consumers.py
import json
import base64
import numpy as np
import cv2
from channels.generic.websocket import AsyncWebsocketConsumer
from PIL import Image
import io
import logging
from .inference.predictor import FaceRecognitionPredictor
from app.users.models import User
from channels.db import database_sync_to_async

logger = logging.getLogger(__name__)

class FaceRecognitionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        logger.info("[WebSocket] Cliente conectado")
        # Cargar predictor solo si no existe
        if not hasattr(self, 'predictor'):
            self.predictor = FaceRecognitionPredictor()
            self.predictor.load_latest_model()
        
        # Caché para optimización
        self.user_cache = {}
        self.last_detection = None
        self.frame_skip_counter = 0
        self.frame_skip_rate = 2  # Procesar 1 de cada 2 frames para mayor velocidad

    async def disconnect(self, close_code):
        logger.info(f"[WebSocket] Cliente desconectado: {close_code}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            image_data = data.get('image')

            if not image_data:
                await self.send(text_data=json.dumps({
                    'success': False,
                    'message': 'No se recibió imagen'
                }))
                return

            # Frame skipping para mayor FPS
            self.frame_skip_counter += 1
            if self.frame_skip_counter < self.frame_skip_rate:
                # Enviar última detección conocida
                if self.last_detection:
                    await self.send(text_data=json.dumps(self.last_detection))
                return
            self.frame_skip_counter = 0

            # Decodificar base64 de forma más eficiente
            image_data_clean = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data_clean)
            
            # Usar numpy directamente en vez de PIL para mayor velocidad
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # OpenCV lee en BGR, convertir a RGB
            if image_array is not None:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                # Fallback a PIL si cv2 falla
                image_pil = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image_pil)
                if len(image_array.shape) == 2:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                elif image_array.shape[2] == 4:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            # Redimensionar imagen para procesamiento más rápido
            height, width = image_array.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Reconocer con suavizado temporal
            result = self.predictor.recognize_face_from_image(
                image_array=image_array,
                use_temporal_smoothing=True
            )

            logger.debug(f"[WebSocket][DEBUG] Resultado del predictor: {result}")

            # IMPORTANTE: Verificar que el resultado tenga la estructura correcta
            if result['success']:
                # Si no hay box en el resultado, agregarlo desde la detección
                if not result.get('box') and result.get('user_id'):
                    # Reutilizar detección del predictor en vez de detectar de nuevo
                    # El predictor ya hizo la detección internamente
                    if hasattr(self.predictor.face_detector, 'last_detection'):
                        detection = self.predictor.face_detector.last_detection
                    else:
                        # Solo detectar si es necesario
                        detection = self.predictor.face_detector.detect_face(image_array=image_array)

                    if detection:
                        # Normalizar coordenadas (0-1)
                        h, w = image_array.shape[:2]
                        box = detection['box']
                        result['box'] = {
                            'x': box[0] / w,
                            'y': box[1] / h,
                            'width': box[2] / w,
                            'height': box[3] / h,
                            'name': None  # Se llenará con el usuario
                        }

            # Obtener datos del usuario con caché
            if result.get('user_id'):
                user_id = result['user_id']
                
                # Verificar caché primero
                if user_id in self.user_cache:
                    user_data = self.user_cache[user_id]
                else:
                    user_data = await self.get_user_data(user_id)
                    self.user_cache[user_id] = user_data
                
                result['user'] = user_data

                # Agregar nombre al box
                if result.get('box') and user_data:
                    result['box']['name'] = user_data.get('full_name') or user_data.get('username')

            # Guardar última detección válida
            if result.get('success'):
                self.last_detection = result

            # Enviar resultado (sin logging para mayor velocidad)
            await self.send(text_data=json.dumps(result))

        except Exception as e:
            logger.error(f"[WebSocket] Error: {str(e)}", exc_info=True)
            await self.send(text_data=json.dumps({
                'success': False,
                'message': f'Error: {str(e)}'
            }))

    @database_sync_to_async
    def get_user_data(self, user_id):
        try:
            user = User.objects.get(id=user_id)
            return {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.get_full_name() or user.username
            }
        except User.DoesNotExist:
            return None
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, callbacks
import numpy as np
from pathlib import Path
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import logging
from datetime import datetime

from ..preprocessing.detector import FaceDetector
from ..preprocessing.transformer import FaceTransformer
from .dataset_manager import FaceDatasetManager

logger = logging.getLogger(__name__)


class FaceCNNModel:
    """
    Arquitectura de CNN personalizada para reconocimiento facial.
    Modelo basado en capas convolucionales para extraer características faciales.
    """
    @staticmethod
    def residual_block(x, filters, name_prefix):
        """Bloque residual para mejor gradiente y profundidad"""
        shortcut = x
        
        # Primera convolución
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        
        # Segunda convolución
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        
        # Ajustar shortcut si es necesario
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same', name=f'{name_prefix}_shortcut')(shortcut)
        
        # Conexión residual
        x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
        
        return x
    
    @staticmethod
    def build_model(input_shape=(160, 160, 3), num_classes=10, embedding_size=128):
        # Input layer
        inputs = layers.Input(shape=input_shape, name='input_image')
        
        # Stem: Convolución inicial
        x = layers.Conv2D(32, (7, 7), strides=2, padding='same', name='stem_conv')(inputs)
        x = layers.BatchNormalization(name='stem_bn')(x)
        x = layers.Activation('relu', name='stem_relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='stem_pool')(x)
        
        # Bloque 1: 64 filtros
        x = FaceCNNModel.residual_block(x, 64, 'block1_res1')
        x = FaceCNNModel.residual_block(x, 64, 'block1_res2')
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.3, name='dropout1')(x)
        
        # Bloque 2: 128 filtros
        x = FaceCNNModel.residual_block(x, 128, 'block2_res1')
        x = FaceCNNModel.residual_block(x, 128, 'block2_res2')
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        
        # Bloque 3: 256 filtros
        x = FaceCNNModel.residual_block(x, 256, 'block3_res1')
        x = FaceCNNModel.residual_block(x, 256, 'block3_res2')
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(0.4, name='dropout3')(x)
        
        # Global Average Pooling en lugar de Flatten
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        # Capas densas con L2 regularization
        x = layers.Dense(512, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        name='fc1')(x)
        x = layers.BatchNormalization(name='bn_fc1')(x)
        x = layers.Dropout(0.5, name='dropout_fc1')(x)
        
        # Embedding layer con normalización L2
        embedding = layers.Dense(embedding_size, activation=None, 
                                kernel_regularizer=keras.regularizers.l2(0.001),
                                name='embedding')(x)
        embedding = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), 
                                 name='embedding_normalized')(embedding)
        x = layers.BatchNormalization(name='bn_embedding')(embedding)
        x = layers.Dropout(0.5, name='dropout_embedding')(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        # Crear modelo
        model = models.Model(inputs=inputs, outputs=outputs, name='FaceRecognitionCNN')
        
        return model
    
    @staticmethod
    def build_embedding_model(base_model):
        embedding_layer = base_model.get_layer('embedding')
        embedding_model = models.Model(
            inputs=base_model.input,
            outputs=embedding_layer.output,
            name='FaceEmbeddingModel'
        )
        
        return embedding_model


class FaceRecognitionTrainer:
    """
    Entrenador para el modelo de reconocimiento facial.
    Gestiona el proceso completo de entrenamiento.
    """

    def __init__(self, model_name='face_recognition_model'):
        self.model_name = model_name
        self.model = None
        self.embedding_model = None
        self.history = None
        self.label_encoder = LabelEncoder()
        self.user_id_mapping = None
        
        # Componentes
        self.dataset_manager = FaceDatasetManager()
        self.face_transformer = FaceTransformer()
        
        # Directorios
        self.models_dir = Path(settings.MODELS_STORAGE_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FaceRecognitionTrainer inicializado: {model_name}")
    
    def prepare_data(self, test_size=0.2, min_images_per_user=5):
        logger.info("Preparando datos de entrenamiento...")
        
        # Cargar datos del dataset
        X, y, user_id_mapping = self.dataset_manager.prepare_training_data(
            min_images_per_user=min_images_per_user
        )
        
        if X is None:
            raise ValueError("No hay suficientes datos para entrenamiento")
        
        self.user_id_mapping = user_id_mapping
        
        # Preprocesar imágenes
        X_processed = self.face_transformer.batch_preprocess(X)
        
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_encoded,
            test_size=test_size,
            stratify=y_encoded,
            random_state=42
        )
        
        logger.info(
            f"Datos preparados - Train: {len(X_train)}, Val: {len(X_val)}, "
            f"Clases: {len(np.unique(y_encoded))}"
        )
        
        return X_train, X_val, y_train, y_val
    
    def build_and_compile_model(self, num_classes, input_shape=(160, 160, 3)):
        logger.info(f"Construyendo modelo con {num_classes} clases...")
        
        self.model = FaceCNNModel.build_model(
            input_shape=input_shape,
            num_classes=num_classes
        )
        
        # Compilar modelo
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Modelo construido y compilado")
        logger.info(f"Total de parámetros: {self.model.count_params():,}")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        logger.info(f"Iniciando entrenamiento - {epochs} épocas, batch_size={batch_size}")
        
        # Callbacks
        checkpoint_path = self.models_dir / f"{self.model_name}_checkpoint.weights.h5"
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Entrenamiento completado")
        
        return self.history
    
    def save_model(self, version='1.0'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{self.model_name}_v{version}_{timestamp}.keras"
        model_path = self.models_dir / model_filename
        
        # Guardar modelo completo
        self.model.save(str(model_path))
        
        # Crear modelo de embeddings
        self.embedding_model = FaceCNNModel.build_embedding_model(self.model)
        embedding_filename = f"{self.model_name}_embedding_v{version}_{timestamp}.keras"
        embedding_path = self.models_dir / embedding_filename
        self.embedding_model.save(str(embedding_path))
        
        # Guardar metadata
        metadata = {
            'model_name': self.model_name,
            'version': version,
            'timestamp': timestamp,
            'model_file': model_filename,
            'embedding_file': embedding_filename,
            'user_id_mapping': self.user_id_mapping,
            'num_classes': len(self.user_id_mapping),
            'input_shape': list(self.model.input_shape[1:]),
            'total_parameters': int(self.model.count_params()),
        }
        
        if self.history:
            metadata['training'] = {
                'final_accuracy': float(self.history.history['accuracy'][-1]),
                'final_loss': float(self.history.history['loss'][-1]),
                'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'epochs_trained': len(self.history.history['loss'])
            }
        
        metadata_filename = f"{self.model_name}_metadata_v{version}_{timestamp}.json"
        metadata_path = self.models_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modelo guardado: {model_path}")
        logger.info(f"Modelo de embeddings guardado: {embedding_path}")
        logger.info(f"Metadata guardada: {metadata_path}")
        
        return {
            'model_path': str(model_path),
            'embedding_path': str(embedding_path),
            'metadata_path': str(metadata_path),
            'metadata': metadata
        }
    
    def full_training_pipeline(self, epochs=50, batch_size=32, version='1.0'):

        # 1. Preparar datos
        X_train, X_val, y_train, y_val = self.prepare_data()
        
        # 2. Construir modelo
        num_classes = len(np.unique(y_train))
        self.build_and_compile_model(num_classes)
        
        # 3. Entrenar
        self.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        # 4. Guardar
        return self.save_model(version)

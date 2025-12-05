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

from ..preprocessing.mfcc_extractor import MFCCExtractor
from ..preprocessing.audio_cleaner import AudioCleaner
from .dataset_manager import VoiceDatasetManager

logger = logging.getLogger(__name__)


class VoiceCNNModel:
    """
    Arquitectura de CNN personalizada para reconocimiento de voz.
    Modelo basado en capas convolucionales 1D para procesar características MFCC.
    """
    
    @staticmethod
    def build_model(input_shape=(39, 100), num_classes=10, embedding_size=128):
        """
        Construye modelo CNN para reconocimiento de voz.
        
        Args:
            input_shape: (n_features, n_timesteps) - típicamente (39, 100) para MFCC+deltas
            num_classes: Número de usuarios/clases
            embedding_size: Dimensión del embedding
        """
        inputs = layers.Input(shape=input_shape, name='input_audio')
        
        # Expandir dimensión para Conv1D
        x = layers.Reshape((input_shape[1], input_shape[0]))(inputs)
        
        # Bloque 1
        x = layers.Conv1D(64, 3, padding='same', name='conv1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Activation('relu', name='relu1')(x)
        x = layers.MaxPooling1D(2, name='pool1')(x)
        x = layers.Dropout(0.3, name='dropout1')(x)
        
        # Bloque 2
        x = layers.Conv1D(128, 3, padding='same', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Activation('relu', name='relu2')(x)
        x = layers.MaxPooling1D(2, name='pool2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        
        # Bloque 3
        x = layers.Conv1D(256, 3, padding='same', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.Activation('relu', name='relu3')(x)
        x = layers.MaxPooling1D(2, name='pool3')(x)
        x = layers.Dropout(0.4, name='dropout3')(x)
        
        # LSTM para capturar dependencias temporales
        x = layers.LSTM(128, return_sequences=False, name='lstm')(x)
        x = layers.Dropout(0.4, name='dropout_lstm')(x)
        
        # Capas densas
        x = layers.Dense(256, activation='relu', 
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
        model = models.Model(inputs=inputs, outputs=outputs, name='VoiceRecognitionCNN')
        
        return model
    
    @staticmethod
    def build_embedding_model(base_model):
        """Construye modelo solo para extracción de embeddings."""
        embedding_layer = base_model.get_layer('embedding_normalized')
        embedding_model = models.Model(
            inputs=base_model.input,
            outputs=embedding_layer.output,
            name='VoiceEmbeddingModel'
        )
        
        return embedding_model


class VoiceRecognitionTrainer:
    """
    Entrenador para el modelo de reconocimiento de voz.
    Gestiona el proceso completo de entrenamiento.
    """
    
    def __init__(self, model_name='voice_recognition_model'):
        self.model_name = model_name
        self.model = None
        self.embedding_model = None
        self.history = None
        self.label_encoder = LabelEncoder()
        self.user_id_mapping = None
        
        # Componentes
        self.dataset_manager = VoiceDatasetManager()
        self.mfcc_extractor = MFCCExtractor()
        self.audio_cleaner = AudioCleaner()
        
        # Directorios
        self.models_dir = Path(settings.MODELS_STORAGE_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VoiceRecognitionTrainer inicializado: {model_name}")
    
    def prepare_data(self, test_size=0.2, min_samples_per_user=5, target_timesteps=100):
        """Prepara datos de entrenamiento extrayendo características MFCC."""
        logger.info("Preparando datos de entrenamiento...")
        
        # Cargar datos del dataset
        X_audio, y, user_id_mapping = self.dataset_manager.prepare_training_data(
            min_samples_per_user=min_samples_per_user
        )
        
        if X_audio is None:
            raise ValueError("No hay suficientes datos para entrenamiento")
        
        self.user_id_mapping = user_id_mapping
        
        # Extraer características MFCC de cada audio
        logger.info("Extrayendo características MFCC...")
        X_features = []
        
        for audio in X_audio:
            # Preprocesar audio
            audio_clean = self.audio_cleaner.preprocess_audio(audio_array=audio)
            
            # Extraer MFCCs (13 coefs + 13 deltas + 13 delta-deltas = 39 features)
            mfccs = self.mfcc_extractor.extract_mfcc(
                audio_array=audio_clean,
                delta=True,
                delta_delta=True
            )
            
            if mfccs is not None:
                # Normalizar
                mfccs = self.mfcc_extractor.normalize_features(mfccs)
                
                # Ajustar longitud
                mfccs = self.mfcc_extractor.pad_or_truncate(mfccs, target_timesteps)
                
                X_features.append(mfccs.T)  # Transponer para tener (timesteps, features)
        
        X_features = np.array(X_features)
        
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y_encoded,
            test_size=test_size,
            stratify=y_encoded,
            random_state=42
        )
        
        logger.info(
            f"Datos preparados - Train: {len(X_train)}, Val: {len(X_val)}, "
            f"Clases: {len(np.unique(y_encoded))}, Shape: {X_train.shape}"
        )
        
        return X_train, X_val, y_train, y_val
    
    def build_and_compile_model(self, num_classes, input_shape=(100, 39)):
        """Construye y compila el modelo."""
        logger.info(f"Construyendo modelo con {num_classes} clases...")
        
        # Ajustar shape para el modelo
        model_input_shape = (input_shape[1], input_shape[0])  # (features, timesteps)
        
        self.model = VoiceCNNModel.build_model(
            input_shape=model_input_shape,
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
        """Entrena el modelo."""
        logger.info(f"Iniciando entrenamiento - {epochs} épocas, batch_size={batch_size}")
        
        # Callbacks
        checkpoint_path = self.models_dir / f"{self.model_name}_checkpoint.weights.h5"
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
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
        """Guarda el modelo entrenado."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{self.model_name}_v{version}_{timestamp}.keras"
        model_path = self.models_dir / model_filename
        
        # Guardar modelo completo
        self.model.save(str(model_path))
        
        # Crear modelo de embeddings
        self.embedding_model = VoiceCNNModel.build_embedding_model(self.model)
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
        """Pipeline completo de entrenamiento."""
        
        # 1. Preparar datos
        X_train, X_val, y_train, y_val = self.prepare_data()
        
        # 2. Construir modelo
        num_classes = len(np.unique(y_train))
        self.build_and_compile_model(num_classes, input_shape=X_train.shape[1:])
        
        # 3. Entrenar
        self.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        # 4. Guardar
        return self.save_model(version)

import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

def entrenar_y_guardar(csv_path='Customer-Churn-Records.csv'):
    print("Cargando dataset...")
    df = pd.read_csv(csv_path)
    
    # 1. Definir percentiles de saldo para segmentación
    saldos = df['Balance']
    p33 = np.percentile(saldos, 33)
    p66 = np.percentile(saldos, 66)
    
    def segmentar_saldo(balance):
        if balance <= p33:
            return 'Bajo'
        elif balance <= p66:
            return 'Medio'
        else:
            return 'Superior'
            
    df['Segmento_Saldo'] = df['Balance'].apply(segmentar_saldo)
    
    X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited']
    
    # 2. Pipeline de Preprocesamiento
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned']
    cat_cols = ['Geography', 'Gender', 'Card Type', 'Segmento_Saldo']
    binary_cols = ['HasCrCard', 'IsActiveMember', 'Complain']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )
    
    # Division de conjuntos
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full)
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_features + binary_cols
    
    # 3. Pesos de clases
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[i]: weights[i] for i in range(len(classes))}
    
    # 4. Red Neuronal
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_proc.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    print("Iniciando entrenamiento del Modelo Predictivo...")
    model.fit(
        X_train_proc, y_train,
        validation_data=(X_val_proc, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Guardar modelo y transformadores
    os.makedirs('modelo_assets', exist_ok=True)
    model.save('modelo_assets/deep_retention_model.keras')
    
    assets = {
        'preprocessor': preprocessor,
        'p33': p33,
        'p66': p66,
        'feature_names': feature_names
    }
    with open('modelo_assets/preprocesador_assets.pkl', 'wb') as f:
        pickle.dump(assets, f)
    print("¡Re-entrenamiento finalizado con éxito! Todos los artefactos fueron actualizados.")

if __name__ == '__main__':
    entrenar_y_guardar()

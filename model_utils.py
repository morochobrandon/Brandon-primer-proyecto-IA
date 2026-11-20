import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

class DeepRetentionSystem:
    def __init__(self, model_path='modelo_assets/deep_retention_model.keras', assets_path='modelo_assets/preprocesador_assets.pkl'):
        # Cargar modelo Keras
        self.model = load_model(model_path)
        
        # Cargar preprocesador y parámetros de saldo
        with open(assets_path, 'rb') as f:
            assets = pickle.load(f)
        self.preprocessor = assets['preprocessor']
        self.p33 = assets['p33']
        self.p66 = assets['p66']
        self.feature_names = assets['feature_names']
        
        # Parámetros financieros
        self.COST_A = 200.0
        self.EFF_A = 0.85
        self.MARGIN_A = 0.10
        
        self.COST_B = 50.0
        self.EFF_B = 0.70
        self.MARGIN_B = 0.08

    def segmentar_saldo(self, balance):
        if balance <= self.p33:
            return 'Bajo'
        elif balance <= self.p66:
            return 'Medio'
        else:
            return 'Superior'

    def predecir_cliente(self, datos_cliente_dict):
        df_cliente = pd.DataFrame([datos_cliente_dict])
        df_cliente['Segmento_Saldo'] = df_cliente['Balance'].apply(self.segmentar_saldo)
        
        # Transformar usando el pipeline guardado
        datos_proc = self.preprocessor.transform(df_cliente)
        
        # Predecir probabilidad con la red neuronal
        prob = float(self.model.predict(datos_proc, verbose=0)[0][0])
        
        saldo_cat = df_cliente.loc[0, 'Segmento_Saldo']
        balance = df_cliente.loc[0, 'Balance']
        
        decision = "Sin Acción"
        incentivo = "Mantenimiento Ordinario"
        costo = 0.0
        ganancia_neta_esp = 0.0
        
        if prob >= 0.65:
            if saldo_cat == 'Superior':
                decision = "Retención Prioritaria"
                incentivo = "Gestor VIP y Tarifa Plana"
                costo = self.COST_A
                ganancia_neta_esp = (prob * self.EFF_A * (balance * self.MARGIN_A)) - costo
            elif saldo_cat == 'Medio':
                decision = "Retención Automatizada"
                incentivo = "Bonificación Transaccional"
                costo = self.COST_B
                ganancia_neta_esp = (prob * self.EFF_B * (balance * self.MARGIN_B)) - costo
            else:
                decision = "Salida Permitida"
                incentivo = "Sin intervención"
                costo = 0.0
                ganancia_neta_esp = 0.0
                
        autorizado = (ganancia_neta_esp > 0) and (decision in ["Retención Prioritaria", "Retención Automatizada"])
        decision_final = decision if autorizado else "Salida Permitida / Sin Intervención"
        costo_final = costo if autorizado else 0.0
        
        return {
            'probabilidad_fuga': prob,
            'segmento_saldo': saldo_cat,
            'decision_final': decision_final,
            'incentivo': incentivo if autorizado else 'Ninguno',
            'costo_aplicado': costo_final,
            'ganancia_esperada': ganancia_neta_esp if autorizado else 0.0
        }

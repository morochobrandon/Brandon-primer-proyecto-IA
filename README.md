# Deep Retention: Sistema Predictivo de Retención Selectiva de Clientes 📊🤖

Este repositorio contiene la solución lista para producción de **"Deep Retention"**, un sistema inteligente diseñado para optimizar el margen operativo de las entidades bancarias. En lugar de aplicar esfuerzos de retención de forma indiscriminada, la plataforma utiliza una arquitectura de **Redes Neuronales Profundas (MLP)** para estimar el riesgo de fuga individual de los clientes y aplica un **Filtro de Rentabilidad Financiera Neta** para autorizar únicamente las campañas comerciales con un retorno esperado estrictamente positivo.

## 🚀 Características Clave
- **Modelo MLP Integrado:** Red Neuronal multicapa optimizada para manejar desbalance de clases.
- **Filtro de Decisión Financiera Neta:** Regla de negocio automatizada que asegura que el beneficio comercial ($10\%$ o $8\%$ del balance por tasa de efectividad) supere los costos directos de campaña ($200 USD o $50 USD).
- **Explicabilidad Diagnóstica:** Integración de valores de interpretabilidad mediante **SHAP** para conocer de forma transparente las variables clave de deserción.
- **Interfaz Web Interactiva:** Portal completo y funcional desarrollado en **Gradio** para las consultas rápidas del personal en sucursales.

---

## 📊 Reporte Ejecutivo de Negocio (Resultados de Test)
- **Clientes Analizados:** 1,500
- **Campañas Rentables Autorizadas:** 223
- **Inversión de Campaña Requerida:** `$30,950.00 USD`
- **Balance de Cartera Salvado:** `$22,174,333.59 USD`
- **Retorno Financiero Neto Recuperado:** `$2,094,372.38 USD`
- **Retorno sobre la Inversión (ROI):** **`6,666.95%`**

---

## 📂 Estructura del Repositorio
```text
├── modelo_assets/
│   ├── deep_retention_model.keras       # Pesos guardados del modelo neural
│   └── preprocesador_assets.pkl        # Pipeline del preprocesador serializado
├── requirements.txt                    # Dependencias del entorno
├── model_utils.py                      # Clase de inferencia autónoma de producción
├── app.py                              # Aplicación interactiva (Gradio)
├── train.py                            # Código de entrenamiento para reproducibilidad
└── README.md                           # Documentación principal del sistema
```

---

## 🔧 Instrucciones de Instalación y Uso

### 1. Clonar el Repositorio
```bash
git clone https://github.com/TU-USUARIO/deep-retention.git
cd deep-retention
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Lanzar la Aplicación Interactiva (Gradio)
```bash
python app.py
```

La aplicación generará un enlace local (`http://127.0.0.1:7860`) y una URL pública temporal para poder acceder y simular de inmediato.

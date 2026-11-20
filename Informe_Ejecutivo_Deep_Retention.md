# INFORME EJECUTIVO DE NEGOCIO: SISTEMA "DEEP RETENTION"

**A:** Junta Directiva del Banco  
**De:** Dirección de Estrategia y Rentabilidad de Clientes  
**Asunto:** Resultados de Implementación y Simulación Financiera del Modelo Predictivo  
**Estado:** Listo para Producción  

---

## 1. Resumen Ejecutivo y Retorno Financiero (ROI)
La retención indiscriminada de clientes ha sido sustituida exitosamente por el motor de decisión selectivo **"Deep Retention"**. Mediante el uso de Inteligencia Artificial (Redes Neuronales), ahora es posible predecir con precisión la probabilidad de fuga e intervenir exclusivamente a aquellos clientes cuya retención genere una rentabilidad neta positiva demostrable para la institución.

* **Clientes analizados en el grupo de prueba (Test):** 1,500 clientes.
* **Clientes que requieren intervención autorizada:** 223 clientes (Segmentos A y B con Ganancia Neta Esperada > 0).
* **Inversión Total en Campañas de Retención:** **$30,950.00 USD**.
* **Balance de Cartera Salvado Estimado:** **$22,174,333.59 USD** (Ajustado por el factor de efectividad de las campañas).
* **Retorno Financiero Neto de la Cartera Salvada (Margen Comercial Recuperado):** **$2,094,372.38 USD**.
* **Retorno Sobre la Inversión (ROI):** **6,666.95%**.

---

## 2. Arquitectura de la Solución y Diagnóstico de Fuga
* **Configuración de la Red Neuronal:**  
  Se diseñó una Red Neuronal Profunda Multicapa (MLP) de 3 capas densas usando Keras/TensorFlow. La arquitectura procesa 23 variables preprocesadas de entrada.  
  - Capa de Entrada y Capa Oculta 1: 128 neuronas (Activación Relu) + Batch Normalization + Dropout (30%) para evitar sobreajuste.  
  - Capa Oculta 2: 64 neuronas (Activación Relu) + Batch Normalization + Dropout (20%).  
  - Capa de Salida: 1 neurona con función de activación Sigmoide (salida continua entre 0.00 y 1.00 de probabilidad de fuga).
  - Optimizador: Adam con pérdida de entropía cruzada binaria (binary_crossentropy).

* **Métricas de Diagnóstico Técnico:**  
  - **ROC-AUC:** **0.9967**  
  - **Precision-Recall AUC:** **0.9968**  
  - **Precisión de Calibración:** **100.00%**  

* **Factores Determinantes de Abandono (Identificados mediante SHAP):**  
  1. **Complain (Reclamaciones):** Principal disparador predictivo. Clientes con quejas activas poseen más de un 90% de riesgo inminente de fuga.
  2. **Age (Edad):** Los clientes en rangos de edad avanzada muestran mayor propensión natural al abandono transaccional.
  3. **IsActiveMember (Miembro Activo):** La falta de actividad frecuente en canales digitales acelera los eventos de deserción.

---

## 3. Matriz de Decisión Comercial y Rentabilidad

| Segmento | Probabilidad Red | Valor Patrimonial | Decisión del Sistema | Incentivo Asignado | Costo ($) | Margen Neto Recuperado Proyectado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Alto Valor / Alto Riesgo** | >= 0.65 | Superior (> $117,303.18) | **Retención Prioritaria** | Gestor VIP y Tarifa Plana | $200.00 | (Prob * 0.85 * (Balance * 0.10)) - $200.00 |
| **B: Medio / Riesgo Moderado** | >= 0.65 | Medio (Hasta $117,303.18) | **Retención Automatizada** | Bonificación Transaccional | $50.00 | (Prob * 0.70 * (Balance * 0.08)) - $50.00 |
| **C: No Rentable** | >= 0.65 | Bajo / Deficitario ($0.00) | **Salida Permitida** | Sin intervención | $0.00 | $0.00 (Ahorro Operativo) |
| **D: Estable** | < 0.30 | Cualquier Saldo | **Sin Acción Requerida** | Mantenimiento Ordinario | $0.00 | Estable |

---

## 4. Operación de la Aplicación en Sucursales
* **Procedimiento del Gestor de Cuentas:** El gestor ingresa los datos del cliente en la aplicación Gradio. El sistema ejecuta el análisis individual, estima el riesgo de fuga mediante la red y evalúa automáticamente el filtro de rentabilidad neta esperada, recomendando instantáneamente la mejor acción comercial.
* **Mantenimiento del Modelo:** Se establece una frecuencia de reentrenamiento programada mensual con nuevos registros transaccionales para recalibrar los pesos de la red.

import gradio as gr
from model_utils import DeepRetentionSystem

# Inicializar motor de inferencia
sistema = DeepRetentionSystem()

def interfaz_gradio(credit_score, geography, gender, age, tenure, balance, num_products, estimated_salary, has_credit_card, is_active_member, complain, card_type, satisfaction_score, points_earned):
    datos = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Complain': int(complain),
        'Satisfaction Score': satisfaction_score,
        'Card Type': card_type,
        'Point Earned': points_earned
    }
    
    res = sistema.predecir_cliente(datos)
    
    retorno_str = f"${res['ganancia_esperada']:,.2f} USD" if res['ganancia_esperada'] > 0 else "$0.00 USD (Inversión no rentable u optimizada)"
    
    res_texto = f"""
    ### RESULTADO DEL ANÁLISIS INDIVIDUAL ("Deep Retention")
    * **Riesgo Estimado de Fuga:** {res['probabilidad_fuga'] * 100:.2f}%
    * **Segmento de Saldo Patrimonial:** {res['segmento_saldo']} (Balance: ${balance:,.2f} USD)
    * **Estatus de Intervención:** **{res['decision_final']}**
    * **Incentivo Propuesto:** {res['incentivo']}
    * **Costo Directo Aplicado:** ${res['costo_aplicado']:,.2f} USD
    * **Retorno Financiero Esperado Proyectado:** {retorno_str}
    """
    return res_texto

# Definición de interfaz
app = gr.Interface(
    fn=interfaz_gradio,
    inputs=[
        gr.Slider(300, 850, value=600, label="Puntaje Crediticio (CreditScore)"),
        gr.Dropdown(['France', 'Spain', 'Germany'], value='France', label="País de Residencia"),
        gr.Dropdown(['Female', 'Male'], value='Female', label="Género"),
        gr.Slider(18, 100, value=40, label="Edad del Cliente"),
        gr.Slider(0, 10, value=5, label="Antigüedad (Tenure - Años)"),
        gr.Number(value=85000.0, label="Saldo en Cuenta (Balance USD)"),
        gr.Slider(1, 4, value=2, label="Número de Productos contratados"),
        gr.Number(value=50000.0, label="Salario Estimado (EstimatedSalary USD)"),
        gr.Checkbox(value=True, label="¿Tiene Tarjeta de Crédito?"),
        gr.Checkbox(value=True, label="¿Es Miembro Activo?"),
        gr.Checkbox(value=False, label="¿Tiene Reclamaciones Activas (Complain)?"),
        gr.Dropdown(['SILVER', 'GOLD', 'PLATINUM', 'DIAMOND'], value='GOLD', label="Tipo de Tarjeta (Card Type)"),
        gr.Slider(1, 5, value=4, label="Puntaje de Satisfacción (Satisfaction Score)"),
        gr.Number(value=500, label="Puntos Acumulados (Points Earned)")
    ],
    outputs=gr.Markdown(label="Resolución de Retención"),
    title="Sistema Predictivo de Decisiones 'Deep Retention'",
    description="Interfaz unificada para gestores bancarios de sucursales. Automatización basada en Redes Neuronales y Filtro Financiero de Beneficios Neta."
)

if __name__ == '__main__':
    app.launch(share=True)

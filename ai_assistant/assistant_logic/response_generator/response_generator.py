# response_generator.py

class ResponseGenerator:
    def __init__(self, language_model, response_processor, response_selector):
        """
        Inicializa el generador de respuestas con los componentes necesarios.

        :param language_model: Instancia de LanguageModel para generar texto basado en la intención.
        :param response_processor: Instancia de ResponseProcessor para ajustar tono y complejidad.
        :param response_selector: Instancia de ResponseSelector para elegir la mejor respuesta.
        """
        self.language_model = language_model
        self.response_processor = response_processor
        self.response_selector = response_selector

    def generate_response(self, intent, keywords, context):
        """
        Genera una respuesta final basada en la intención, contexto y palabras clave.

        :param intent: Intención detectada (str).
        :param keywords: Palabras clave extraídas de la entrada del usuario (list).
        :param context: Contexto actual de la conversación (dict).
        :return: Respuesta final generada (str).
        """
        # 1. Usa el modelo de lenguaje para generar respuestas preliminares
        raw_responses = self.language_model.generate(intent, keywords, context)
        print(f"[DEBUG] Respuestas generadas por el modelo: {raw_responses}")

        # 2. Procesa las respuestas para ajustar el tono y la complejidad
        processed_responses = [
            self.response_processor.process_response(response) for response in raw_responses
        ]
        print(f"[DEBUG] Respuestas procesadas: {processed_responses}")

        # 3. Selecciona la mejor respuesta basada en el contexto
        final_response = self.response_selector.select_best_response(
            processed_responses, context)
        print(f"[DEBUG] Respuesta seleccionada: {final_response}")

        return final_response

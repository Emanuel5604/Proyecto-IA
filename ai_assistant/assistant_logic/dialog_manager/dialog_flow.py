# dialog_flow.py

class DialogFlow:
    def __init__(self, context_manager, emotion_handler, intent_recognizer, response_generator):
        """
        Inicializa el flujo de diálogo con los componentes necesarios.

        :param context_manager: Instancia de ContextManager para gestionar el contexto.
        :param emotion_handler: Instancia de EmotionHandler para manejar emociones.
        :param intent_recognizer: Instancia de IntentRecognizer para detectar intenciones.
        :param response_generator: Instancia de ResponseGenerator para generar respuestas.
        """
        self.context_manager = context_manager
        self.emotion_handler = emotion_handler
        self.intent_recognizer = intent_recognizer
        self.response_generator = response_generator

    def handle_user_input(self, user_input, detected_emotion):
        """
        Gestiona el flujo de la conversación basado en la entrada del usuario y las emociones detectadas.

        :param user_input: Entrada de texto del usuario.
        :param detected_emotion: Emoción detectada (str).
        :return: Respuesta generada para el usuario (str).
        """
        # 1. Reconocer intención del usuario
        intent = self.intent_recognizer.predict_intent(user_input)
        print(f"[INFO] Intención detectada: {intent}")

        # 2. Extraer palabras clave para enriquecer el contexto
        keywords = self.intent_recognizer.extract_keywords(user_input)
        self.context_manager.update_context(keywords)
        print(f"[INFO] Contexto actualizado con palabras clave: {keywords}")

        # 3. Procesar la emoción detectada
        emotion_adjustments = self.emotion_handler.process_emotion(
            detected_emotion)
        print(f"[INFO] Ajustes basados en emoción: {emotion_adjustments}")

        # 4. Generar una respuesta preliminar
        preliminary_response = self.response_generator.generate_response(
            intent, keywords, self.context_manager.get_context())
        print(f"[INFO] Respuesta preliminar generada: {preliminary_response}")

        # 5. Ajustar la respuesta según la emoción
        final_response = self.emotion_handler.adjust_response(
            preliminary_response, detected_emotion)
        print(f"[INFO] Respuesta final ajustada: {final_response}")

        return final_response

    def reset_conversation(self):
        """
        Resetea el contexto y el flujo de diálogo.
        """
        self.context_manager.reset_context()
        print("[INFO] Contexto reseteado. Nueva conversación iniciada.")

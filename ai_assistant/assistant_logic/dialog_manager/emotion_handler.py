# emotion_handler.py

class EmotionHandler:
    def __init__(self):
        """
        Inicializa el manejador de emociones con configuraciones predefinidas
        para ajustar el tono y la estrategia del diálogo según las emociones detectadas.
        """
        # Mapear emociones a ajustes en el flujo del diálogo
        self.emotion_response_map = {
            "happy": {"tone": "positive", "strategy": "encourage"},
            "sad": {"tone": "empathetic", "strategy": "console"},
            "angry": {"tone": "calm", "strategy": "de-escalate"},
            "neutral": {"tone": "neutral", "strategy": "inform"},
            "fearful": {"tone": "reassuring", "strategy": "comfort"},
            "surprised": {"tone": "curious", "strategy": "inquire"}
        }
        # Almacenar la última emoción procesada
        self.last_emotion = None

    def process_emotion(self, emotion):
        """
        Procesa la emoción detectada y selecciona los ajustes de tono y estrategia adecuados.

        :param emotion: Emoción detectada (str), por ejemplo: 'happy', 'sad', etc.
        :return: Diccionario con el tono y la estrategia para ajustar la conversación.
        """
        self.last_emotion = emotion
        return self.emotion_response_map.get(emotion, {"tone": "neutral", "strategy": "inform"})

    def adjust_response(self, response, emotion):
        """
        Ajusta la respuesta del asistente según el tono derivado de la emoción.

        :param response: Texto de la respuesta generada.
        :param emotion: Emoción detectada.
        :return: Respuesta ajustada con el tono adecuado.
        """
        emotion_settings = self.process_emotion(emotion)
        tone = emotion_settings["tone"]

        # Lógica simplificada para ajustar el tono de la respuesta.
        if tone == "positive":
            return f"¡Eso suena genial! {response}"
        elif tone == "empathetic":
            return f"Lamento escuchar eso. {response}"
        elif tone == "calm":
            return f"Entiendo tu punto. {response}"
        elif tone == "reassuring":
            return f"No te preocupes, estoy aquí para ayudarte. {response}"
        elif tone == "curious":
            return f"¡Qué interesante! {response}"
        else:
            return response

    def get_last_emotion(self):
        """
        Devuelve la última emoción procesada.

        :return: Última emoción detectada.
        """
        return self.last_emotion

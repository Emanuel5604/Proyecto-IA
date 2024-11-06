import random
from textblob import TextBlob


class ResponseSelector:
    def __init__(self, diversity_threshold=0.5):
        """
        Inicializa el selector de respuestas, que selecciona la respuesta más adecuada
        en función de la emoción y el contexto.

        :param diversity_threshold: Controla la diversidad en la selección de respuestas.
        """
        self.diversity_threshold = diversity_threshold

    def filter_by_emotion(self, responses, emotion):
        """
        Filtra las respuestas en función de la emoción del usuario.

        :param responses: Lista de respuestas generadas por el modelo de lenguaje.
        :param emotion: Emoción detectada en el usuario.
        :return: Lista de respuestas que se ajustan a la emoción.
        """
        emotion_responses = []

        for response in responses:
            # Condiciones de ejemplo para adaptar el contenido según la emoción
            if emotion == "feliz" and "feliz" in response.lower():
                emotion_responses.append(response)
            elif emotion == "triste" and any(word in response.lower() for word in ["lamento", "entiendo"]):
                emotion_responses.append(response)
            elif emotion == "enojado" and "comprendo" in response.lower():
                emotion_responses.append(response)

        # Devuelve todas si ninguna coincide
        return emotion_responses if emotion_responses else responses

    def rank_by_relevance(self, responses, context):
        """
        Ordena las respuestas según su relevancia en relación con el contexto.

        :param responses: Lista de respuestas.
        :param context: Contexto de la conversación actual.
        :return: Lista de respuestas ordenadas por relevancia.
        """
        context_blob = TextBlob(context)
        relevance_scores = []

        for response in responses:
            response_blob = TextBlob(response)
            relevance_score = context_blob.similarity(response_blob)
            relevance_scores.append((response, relevance_score))

        # Ordena por relevancia descendente
        ranked_responses = sorted(
            relevance_scores, key=lambda x: x[1], reverse=True)
        return [response for response, score in ranked_responses]

    def select_diverse_response(self, responses):
        """
        Selecciona una respuesta que no sea repetitiva, manteniendo la diversidad en la conversación.

        :param responses: Lista de respuestas.
        :return: Respuesta seleccionada de forma diversa.
        """
        if len(responses) > 1 and random.random() < self.diversity_threshold:
            # Evita la primera para mayor diversidad
            return random.choice(responses[1:])
        return responses[0]

    def select_response(self, responses, emotion=None, context=""):
        """
        Selecciona la respuesta final en función de la emoción y el contexto.

        :param responses: Lista de respuestas generadas por el modelo de lenguaje.
        :param emotion: Emoción detectada en el usuario.
        :param context: Contexto de la conversación actual.
        :return: Respuesta seleccionada.
        """
        # Filtra las respuestas en función de la emoción
        filtered_responses = self.filter_by_emotion(responses, emotion)

        # Ordena por relevancia en relación al contexto
        ranked_responses = self.rank_by_relevance(filtered_responses, context)

        # Selecciona una respuesta diversa
        selected_response = self.select_diverse_response(ranked_responses)

        return selected_response

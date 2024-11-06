import re
from textblob import TextBlob  # type: ignore


class ResponseProcessor:
    def __init__(self, formality_level="neutral", max_length=150):
        """
        Inicializa el procesador de respuestas para ajustar el tono y limpiar la respuesta.

        :param formality_level: Nivel de formalidad de la respuesta (por defecto, "neutral").
        :param max_length: Longitud máxima de la respuesta procesada.
        """
        self.formality_level = formality_level
        self.max_length = max_length

    def adjust_tone(self, response, emotion=None):
        """
        Ajusta el tono de la respuesta en función de la emoción del usuario.

        :param response: Respuesta generada por el modelo de lenguaje.
        :param emotion: Emoción del usuario, que influye en el tono de la respuesta.
        :return: Respuesta con el tono ajustado.
        """
        # Modificar la respuesta según la emoción detectada
        if emotion == "enojado":
            response = "Entiendo que te sientes molesto. " + response
        elif emotion == "triste":
            response = "Lamento escuchar eso. " + response
        elif emotion == "feliz":
            response = "¡Me alegra escuchar buenas noticias! " + response

        # Ajuste según el nivel de formalidad
        if self.formality_level == "formal":
            response = response.replace("tú", "usted")
        elif self.formality_level == "informal":
            response = response.replace("usted", "tú")

        return response

    def check_sensitivity(self, response):
        """
        Revisa la respuesta en busca de contenido sensible o inapropiado.

        :param response: Respuesta generada por el modelo de lenguaje.
        :return: Respuesta revisada y ajustada en caso de encontrar contenido sensible.
        """
        # Ejemplo simple de palabras sensibles que se pueden ajustar
        sensitive_words = ["odio", "matar", "violencia"]
        for word in sensitive_words:
            response = re.sub(
                rf"\b{word}\b", "[contenido sensible]", response, flags=re.IGNORECASE)
        return response

    def grammar_check(self, response):
        """
        Realiza una revisión gramatical básica y ajuste del texto.

        :param response: Respuesta generada por el modelo de lenguaje.
        :return: Respuesta con corrección gramatical.
        """
        blob = TextBlob(response)
        corrected_response = str(blob.correct())
        return corrected_response

    def trim_response(self, response):
        """
        Ajusta la longitud de la respuesta para que se ajuste al máximo permitido.

        :param response: Respuesta generada por el modelo de lenguaje.
        :return: Respuesta ajustada a la longitud máxima permitida.
        """
        if len(response) > self.max_length:
            response = response[:self.max_length].rsplit(" ", 1)[0] + "..."
        return response

    def process_response(self, response, emotion=None):
        """
        Procesa la respuesta para ajustarla en tono, contenido y longitud.

        :param response: Respuesta generada por el modelo de lenguaje.
        :param emotion: Emoción del usuario para adaptar el tono.
        :return: Respuesta final procesada.
        """
        response = self.adjust_tone(response, emotion)
        response = self.check_sensitivity(response)
        response = self.grammar_check(response)
        response = self.trim_response(response)
        return response

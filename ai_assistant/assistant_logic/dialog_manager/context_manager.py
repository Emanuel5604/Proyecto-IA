# context_manager.py

class ContextManager:
    def __init__(self):
        # Almacena el historial de la conversación
        self.conversation_history = []
        # Datos de contexto clave (por ejemplo, nombre del usuario, temas discutidos)
        self.context_data = {}
        # Estado actual del diálogo (por ejemplo, tema o intención actual)
        self.current_state = None

    def update_history(self, user_input, assistant_response):
        """
        Actualiza el historial de la conversación con la última entrada del usuario y la respuesta del asistente.

        :param user_input: Mensaje del usuario.
        :param assistant_response: Respuesta generada por el asistente.
        """
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response
        })

    def set_context_data(self, key, value):
        """
        Establece un dato clave en el contexto de la conversación.

        :param key: Clave del dato (por ejemplo, 'nombre', 'tema_actual').
        :param value: Valor del dato.
        """
        self.context_data[key] = value

    def get_context_data(self, key, default=None):
        """
        Recupera un dato clave del contexto de la conversación.

        :param key: Clave del dato.
        :param default: Valor por defecto si la clave no está en el contexto.
        :return: Valor del dato o el valor por defecto.
        """
        return self.context_data.get(key, default)

    def set_current_state(self, state):
        """
        Establece el estado actual de la conversación (por ejemplo, el tema actual o la intención).

        :param state: Estado actual del diálogo.
        """
        self.current_state = state

    def get_current_state(self):
        """
        Recupera el estado actual de la conversación.

        :return: Estado actual del diálogo.
        """
        return self.current_state

    def clear_context(self):
        """
        Limpia el contexto y el historial de la conversación, útil para reiniciar el diálogo.
        """
        self.conversation_history.clear()
        self.context_data.clear()
        self.current_state = None

    def get_conversation_history(self, last_n=5):
        """
        Obtiene los últimos mensajes del historial de la conversación.

        :param last_n: Número de mensajes a obtener desde el final del historial.
        :return: Lista de los últimos 'n' mensajes del historial.
        """
        return self.conversation_history[-last_n:]

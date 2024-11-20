# intent_recognizer.py

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


class IntentRecognizer:
    def __init__(self, model_path=None):
        """
        Inicializa el reconocedor de intenciones. Si se proporciona un modelo, se carga.
        :param model_path: Ruta del modelo preentrenado (opcional).
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer()
        self.intent_model = None

        if model_path:
            self.load_model(model_path)

    def train(self, training_data, labels):
        """
        Entrena el modelo de intenciones con datos proporcionados.

        :param training_data: Lista de oraciones de entrenamiento (strings).
        :param labels: Lista de etiquetas correspondientes a cada oración.
        """
        # Extrae características de las oraciones
        X = self.vectorizer.fit_transform(training_data)
        # Entrena un modelo simple de clasificación
        self.intent_model = LogisticRegression()
        self.intent_model.fit(X, labels)

    def predict_intent(self, user_input):
        """
        Predice la intención de un mensaje del usuario.

        :param user_input: Entrada del usuario como texto (string).
        :return: Etiqueta de intención predicha.
        """
        if not self.intent_model:
            raise ValueError("El modelo de intenciones no está entrenado.")

        # Preprocesa el texto y predice
        processed_input = self.vectorizer.transform([user_input])
        intent = self.intent_model.predict(processed_input)[0]
        return intent

    def extract_keywords(self, user_input):
        """
        Extrae palabras clave del mensaje del usuario utilizando spaCy.

        :param user_input: Entrada del usuario como texto (string).
        :return: Lista de palabras clave.
        """
        doc = self.nlp(user_input)
        keywords = [
            token.text for token in doc if token.is_alpha and not token.is_stop]
        return keywords

    def save_model(self, model_path):
        """
        Guarda el modelo entrenado y el vectorizador.

        :param model_path: Ruta donde se guardará el modelo.
        """
        with open(model_path, 'wb') as f:
            pickle.dump((self.vectorizer, self.intent_model), f)

    def load_model(self, model_path):
        """
        Carga un modelo preentrenado desde un archivo.

        :param model_path: Ruta del archivo del modelo.
        """
        with open(model_path, 'rb') as f:
            self.vectorizer, self.intent_model = pickle.load(f)

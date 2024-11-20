# main.py

from ai_assistant.assistant_logic.response_generator.response_generator import ResponseGenerator
from assistant_logic.response_generator.language_model import LanguageModel
from assistant_logic.response_generator.response_processor import ResponseProcessor
from assistant_logic.response_generator.response_selector import ResponseSelector

from assistant_logic.dialog_manager.context_manager import ContextManager
from assistant_logic.dialog_manager.emotion_handler import EmotionHandler
from assistant_logic.dialog_manager.intent_recognizer import IntentRecognizer
from assistant_logic.dialog_manager.dialog_flow import DialogFlow


def main():
    # Inicialización de componentes del asistente
    print("[INFO] Inicializando los componentes del asistente...")

    # Componentes del Response Generator
    language_model = LanguageModel(model_path="path_to_language_model")
    response_processor = ResponseProcessor(tone="formal", complexity="high")
    response_selector = ResponseSelector()

    # Componentes del Dialog Manager
    context_manager = ContextManager()
    emotion_handler = EmotionHandler()
    intent_recognizer = IntentRecognizer(model_path="path_to_intent_model.pkl")

    # Inicialización del Generador de Respuestas
    response_generator = ResponseGenerator(
        language_model=language_model,
        response_processor=response_processor,
        response_selector=response_selector,
    )

    # Inicialización del flujo de diálogo
    dialog_flow = DialogFlow(
        context_manager=context_manager,
        emotion_handler=emotion_handler,
        intent_recognizer=intent_recognizer,
        response_generator=response_generator,
    )

    print("[INFO] Asistente inicializado correctamente.")

    # Simulación de interacción
    print("\n[Asistente Virtual] Hola, ¿cómo puedo ayudarte hoy?")
    while True:
        try:
            user_input = input("[Usuario] ")
            if user_input.lower() in ["salir", "adiós", "exit", "quit"]:
                print(
                    "[Asistente Virtual] Ha sido un placer hablar contigo. ¡Hasta luego!")
                break

            # Simulación de emoción detectada (esto se conectaría con el módulo Emotion Calculation)
            detected_emotion = "neutral"  # Se puede reemplazar con una emoción detectada real

            # Manejo de la entrada y generación de respuesta
            response = dialog_flow.handle_user_input(
                user_input, detected_emotion)
            print(f"[Asistente Virtual] {response}")

        except KeyboardInterrupt:
            print("\n[Asistente Virtual] Adiós, que tengas un buen día.")
            break


if __name__ == "__main__":
    main()

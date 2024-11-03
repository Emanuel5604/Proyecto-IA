from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class LanguageModel:
    def __init__(self, model_name='gpt2', max_length=100):
        # Cargar el modelo y el tokenizador preentrenados
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def generate_response(self, prompt, emotion=None, context=None):
        """
        Genera una respuesta basada en un prompt y, opcionalmente, en la emoción y el contexto.

        :param prompt: Texto de entrada para generar la respuesta.
        :param emotion: Emoción del usuario que influirá en la respuesta.
        :param context: Contexto de conversación para personalizar la respuesta.
        :return: Respuesta generada.
        """

        # Personalizar el prompt con la emoción y contexto si se proporcionan
        if emotion:
            prompt = f"[Emoción: {emotion}] {prompt}"
        if context:
            prompt = f"[Contexto: {context}] {prompt}"

        # Tokenizar el prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generar la respuesta con el modelo
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decodificar la salida y devolverla como respuesta
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


# Ejemplo de uso de la clase LanguageModel
if __name__ == "__main__":
    lm = LanguageModel()
    prompt = "¿Cómo puedo ayudarte hoy?"
    emotion = "feliz"
    context = "Usuario ha solicitado ayuda con tareas"

    response = lm.generate_response(prompt, emotion=emotion, context=context)
    print(response)

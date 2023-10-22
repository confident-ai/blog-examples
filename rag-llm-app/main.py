import chromadb
from chromadb.utils import embedding_functions
import openai

client = chromadb.Client()
client.heartbeat()

class Retriver:
    def __init__(self):
        pass
    
    def get_retrieval_results(self, input, k):
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key="your-openai-api-key", model_name="text-embedding-ada-002")
        collection = client.get_collection(name="my_collection", embedding_function=openai_ef)
        retrieval_results = collection.query(
            query_texts=[input],
            n_results=k,
        )
        return retrieval_results["documents"][0]
    


class Generator:
    def __init__(self, openai_model="gpt-4"):
        self.openai_model = openai_model
        self.prompt_template = """
            You're a helpful assistant with a thick country accent. Answer the question below and if you don't know the answer, say you don't know.

            {text}
        """

    def generate_response(self, retrieval_results):
        prompts = []
        for result in retrieval_results:
            prompt = self.prompt_template.format(text=result)
            prompts.append(prompt)
        prompts.reverse()

        response = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[{"role": "assistant", "content": prompt} for prompt in prompts],
            temperature=0,
        )

        return response["choices"][0]["message"]["content"]
    

class Chatbot:
    def __init__(self):
        self.retriver = Retriver()
        self.generator = Generator()
    
    def answer(self, input):
        retrieval_results = self.retriver.get_retrieval_results(input)
        return self.generator.generate_response(retrieval_results)
    

# Creating an instance of the Chatbot class
chatbot = Chatbot()

while True:
    user_input = input("You: ")  # Taking user input from the CLI
    response = chatbot.answer(user_input)
    print(f"Chatbot: {response}")
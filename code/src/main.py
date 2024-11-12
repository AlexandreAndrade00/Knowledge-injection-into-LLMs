from plm.plm_builder import PLMBuilder, Device
from oracle import Oracle


def main():
    builder = PLMBuilder()
    builder.select_model("meta-llama/Llama-3.2-3B-Instruct")
    builder.select_device(Device.CPU)
    builder.set_input_formatter(lambda model_input, context: [
        {"role": "system",
         "content": "You are a chatbot with the objective of answering user questions concisely!"},
        {"role": "system", "content": f"Use the following sentences to answer the question\n{context}"},
        {"role": "user", "content": model_input},
    ])
    plm = builder.build()
    oracle = Oracle(plm)
    oracle.set_question("Where Michael Jackson, the singer, born?")
    print(oracle.answer())


if __name__ == "__main__":
    main()

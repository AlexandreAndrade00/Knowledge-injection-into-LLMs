from plm.plm_builder import PLMBuilder


def main():
    builder = PLMBuilder()
    builder.select_model("meta-llama/Llama-3.2-1B")
    plm = builder.build()
    print(plm.inference("Can you tell me all the presidents of the United States of America?"))


if __name__ == "__main__":
    main()

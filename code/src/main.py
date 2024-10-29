from plm.plm_builder import PLMBuilder, Device


def main():
    builder = PLMBuilder()
    builder.select_model("meta-llama/Llama-3.2-3B-Instruct")
    builder.select_device(Device.CPU)
    plm = builder.build()
    print(plm.inference("How many languages do you know?"))


if __name__ == "__main__":
    main()

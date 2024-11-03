from plm.plm_builder import PLMBuilder, Device
from oracle import Oracle


def main():
    builder = PLMBuilder()
    builder.select_model("meta-llama/Llama-3.2-3B-Instruct")
    builder.select_device(Device.CPU)
    plm = builder.build()
    oracle = Oracle(plm)
    print(oracle.answer("Where Michael Jackson, the singer, born?"))


if __name__ == "__main__":
    main()

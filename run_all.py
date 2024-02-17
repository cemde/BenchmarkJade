import os


def run_script(file: str, dataset: str, devices: str):
    os.system(f"python {file} --dataset {dataset} --devices {devices}")


if __name__ == "__main__":
    DEVICES = "0,1,2,3,4,5,6,7"
    DEVICES = "0,"
    print(f"Devices: {DEVICES}")
    
    run_script("train.py", "imagenet", DEVICES)
    run_script("train.py", "dummy", DEVICES)

    run_script("inference.py", "imagenet", DEVICES)
    run_script("inference.py", "dummy", DEVICES)

    run_script("smooth_inference.py", "imagenet", DEVICES)
    run_script("smooth_inference.py", "dummy", DEVICES)

    print("All done!")

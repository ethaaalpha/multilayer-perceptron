import csv
import random
from argparse import ArgumentParser, ArgumentTypeError

def export(filepath: str, content):
    with open(filepath, 'w+') as file:
        wr = csv.writer(file)
        wr.writerows(content)

def split_data(data_path: str, dist: float):
    with open(data_path, 'r') as file:
        data = list(csv.reader(file, delimiter=","))
        random.shuffle(data)
        index = round(len(data) * dist)

        export("training.csv", data[:index])
        export("validation.csv", data[index:])
        print(f"{len(data[:index])} + {len(data[index:])} = {len(data)}")

def dist_parsing(value):
    float_v = float(value)

    if not (0.0 < float_v < 1.0):
        raise ArgumentTypeError("Float value for distribution must be between 0 and 1!")
    else:
        return float_v

def main():
    parser = ArgumentParser()
    parser.add_argument("file", help="the raw dataset to be splitted")
    parser.add_argument("--seed", help="a random int used as random seed for splitting", type=int)
    parser.add_argument("--dist", help="float value defining the training distribution (ex=0.8; for 80% training, 20% validation)", type=dist_parsing, default=0.8)

    args = parser.parse_args()
    if (args.seed):
        random.seed(args.seed)

    split_data(args.file, args.dist)

if __name__ == "__main__":
    main()

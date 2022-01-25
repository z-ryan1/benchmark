import argparse
import csv
import json


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, type=str)
    parser.add_argument("--out", default="./output.csv", type=str)
    temp_args = parser.parse_args()
    return temp_args


def main(args):

    with open(args.file, "r") as fp:
        json_dict = json.load(fp)

    stats_dict = {}

    for b in json_dict['benchmarks']:

        test_or_val, model_name = b['name'].split("[")

        model_name = model_name.replace("]", "")

        model_name = f"{model_name}_{test_or_val}"

        stats_dict[model_name] = b['stats']['median'] * 1000

    keys = list(reversed(sorted(stats_dict)))

    with open(args.out, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(keys)
        writer.writerow(stats_dict[k] for k in keys)

    print("parsing file {} and outputting to file {}".format(
        args.file, args.out))


if __name__ == "__main__":
    main(prepare_args())

import argparse


def get_yml_name():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "yml_name", metavar="<yml_name>"
    )
    args = parser.parse_args()
    return args.yml_name

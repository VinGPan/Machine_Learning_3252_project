import argparse
import yaml


def read_yml(yml_name):
    with open(yml_name, 'r') as ymlfile:
        configs = yaml.load(ymlfile, Loader=yaml.CLoader)
    return configs


def get_yml_name():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "yml_name", metavar="<yml_name>"
    )
    args = parser.parse_args()
    return args.yml_name

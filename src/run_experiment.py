from src.utils import get_yml_name
from src.s02_prepare_data import prepare_data


def run_experiment(yml_name):
    prepare_data(yml_name, training=True)


if __name__ == '__main__':
    exp_yml_name = get_yml_name()
    run_experiment(exp_yml_name)

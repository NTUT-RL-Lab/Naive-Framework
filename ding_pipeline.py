from config.die_algorithms import die_rainbow, die_r2d2
from coef import Coef


def serial_pipeline(
    config_path: str
):
    cf = Coef(config_path, rlf="ding")
    if cf.algorithm_name == "rainbow":
        die_rainbow.main(config_path)
    elif cf.algorithm_name == "r2d2":
        die_r2d2.main(config_path)


def eval(config_path: str, model_path):
    cf = Coef(config_path, rlf="ding")
    if cf.algorithm_name == "rainbow":
        die_rainbow.eval(config_path, model_path)
    elif cf.algorithm_name == "r2d2":
        die_r2d2.eval(config_path, model_path)

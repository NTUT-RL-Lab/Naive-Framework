from easydict import EasyDict
from not_die_alone import not_alone
from not_die_eval import not_die_eval
dg_rainbow_config = dict(
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=8,
        stop_value=10000000,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
            v_min=-10,
            v_max=10,
            n_atom=51,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            iqn=False,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.05,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, ),
        ),
    ),
)
dg_rainbow_config = EasyDict(dg_rainbow_config)
main_config = dg_rainbow_config
dg_rainbow_create_config = dict(
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
)
dg_rainbow_create_config = EasyDict(
    dg_rainbow_create_config)
create_config = dg_rainbow_create_config


def main(config_path):
    not_alone((main_config, create_config), config_path=config_path)


def eval(config_path, model_path):
    not_die_eval(main_config, create_config,
                 ckpt_path=model_path, config_path=config_path)

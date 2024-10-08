from copy import deepcopy
from easydict import EasyDict
from not_die_alone import not_alone
from not_die_eval import not_die_eval
phoenix_rainbow_config = dict(
    exp_name='dp_20M_rainbow_blend_last_try',
    conf_path='config/dp_blend_DQN.toml',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20000000,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        # load_path='not_die_logs\phoenix10M_rainbow_blend\ckpt\ckpt_best.pth.tar',
        cuda=True,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=[1, 84, 84],
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
phoenix_rainbow_config = EasyDict(phoenix_rainbow_config)
main_config = phoenix_rainbow_config
phoenix_rainbow_create_config = dict(
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
)
phoenix_rainbow_create_config = EasyDict(
    phoenix_rainbow_create_config)
create_config = phoenix_rainbow_create_config


def main():
    not_alone((main_config, create_config), seed=0, max_env_step=20000000)


def eval():
    not_die_eval(main_config, create_config,
                 'not_die_logs\dp_20M_rainbow_blend_last_try_240919_202200\ckpt\ckpt_best.pth.tar')

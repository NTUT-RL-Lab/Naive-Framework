from easydict import EasyDict

pd_config = dict(
    exp_name='test_ding_phoenix_demon_attack',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=8,
        stop_value=10000000000,
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[1, 84, 84],
            action_shape=8,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate_fraction=2.5e-9,
            learning_rate_quantile=0.00005,
            target_update_freq=500,
            ent_coef=0,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=1000000, ),
        ),
    ),
)
pd_config = EasyDict(pd_config)
main_config = pd_config
pd_create_config = dict(
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
pd_create_config = EasyDict(pd_create_config)
create_config = pd_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c phoenix_fqf_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)

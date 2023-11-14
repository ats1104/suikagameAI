from gym.envs.registration import register
print("register")
register(
    id='myenv-v0',
    entry_point='src.main:MyEnv'
)
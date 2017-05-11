from gym.envs.registration import register

register(
    id='hanoi-5discs-v0',
    entry_point='gym_hanoi.envs:Hanoi5Discs'
)
register(
    id='hanoi-10discs-v0',
    entry_point='gym_hanoi.envs:Hanoi10Discs'
)

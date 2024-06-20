from gymnasium.envs.registration import register
register(
    id='AnnulusMove-v0',
    entry_point='air_corridor.d2.scenario:AnnulusMove'
)

register(
    id='AnnulusMove-v1',
    entry_point='air_corridor.d2.scenario:AnnulusMoveV1'
)

# register(
#     id='AnnulusMove-v2',
#     entry_point='air_corridor.d2.scenario:AnnulusMoveV2'
# )

register(
    id='D2Move-v0',
    entry_point='air_corridor.d2.scenario:parallel_env'
)

# register(
#     id='D2Move-v1',
#     entry_point='air_corridor.d2.scenario:D2MoveV1'
# )

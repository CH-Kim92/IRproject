from gym.envs.registration import register

register(
    id="warehouse_grid/GridWorld-v0",
    entry_point="warehouse_grid.envs:GridWorldEnv",
)

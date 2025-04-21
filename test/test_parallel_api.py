from GPE.env.graph_pe import GPE

# from test2 import CustomActionMaskedEnvironment

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = GPE()
    parallel_api_test(env, num_cycles=1_000_000)

    # env = CustomActionMaskedEnvironment()
    # parallel_api_test(env, num_cycles=1_000_000)

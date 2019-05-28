import configwrapper

# import argparse
# parser = argparse.ArgumentParser(prog="main script")
#
# parser.add_argument("--config-env", required=True,
#                    help="environment config (REQUIRED)")
# parser.add_argument("--config-agent", required=True,
#                    help="agent config (REQUIRED)")
#
# args = parser.parse_args()


with open('configs/envs/fetch.json', 'r') as myfile:
    config_env = myfile.read()

with open('configs/agents/fetch/doubledqn.json', 'r') as myfile:
    config_agent = myfile.read()

configwrapper.wrapper(config_env, config_agent)
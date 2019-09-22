import aggregator.aggregator as aggregator

#path = "/home/gcideron/out_texther/out_texther/rnn_Fetch_12x12_all_missions/rnn-Fetch-12x12-N2-C6-SE5-SI5-O10/"
path = "/home/gcideron/out_texther/out_texther/noisy_her/noisy-her-rnn-Fetch-12x12-N2-C6-SE5-SI5-O10/"
#aggregator.wrapper(path+"dueling-double-dqn-her", output="csv")
#aggregator.wrapper(path+"dueling-double-dqn-no-expert", output="csv")
#aggregator.wrapper(path+"dueling-double-dqn-expert-to-learn-rnn", output="csv")
aggregator.wrapper(path+"dueling-double-dqn-noisy-her-noise-random-0.2", output="csv")
aggregator.wrapper(path+"dueling-double-dqn-noisy-her-noise-random-0.3", output="csv")
aggregator.wrapper(path+"dueling-double-dqn-noisy-her-noise-random-0.4", output="csv")
aggregator.wrapper(path+"dueling-double-dqn-noisy-her-noise-random-0.5", output="csv")
aggregator.wrapper(path+"dueling-double-dqn-noisy-her-noise-random-0.6", output="csv")
aggregator.wrapper(path+"dueling-double-dqn-noisy-her-noise-random-0.8", output="csv")
aggregator.wrapper(path+"dueling-double-dqn-noisy-her-noise-random-0.9", output="csv")



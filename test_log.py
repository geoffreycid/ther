#with open("out_pred_language/sentences.csv", "a") as log:
#    log.write("{},{},{}\n".format("index", "prediction", "truth"))
#with open("out_pred_language/sentences.csv", "a") as log:
#    for i in range(10):
#        log.write("{},{},{}\n".format(i, "get a a a", "get a red ball"))

#import csv
#with open('testlogsecond.csv', 'w', newline='') as outcsv:
#    writer = csv.writer(outcsv)
#    writer.writerow(["Date", "temperature 1", "Temperature 2"])

#with open('testlogsecond.csv', 'w', newline='') as outcsv:
#    writer = csv.writer(outcsv)
#    for i in range(10):
#        #writer.writerow("{},{},{}".format(i, "get a a a", "get a red ball"))
#        writer.writerow([i, "get a a a", "get a red ball"])


import aggregator.aggregator as aggregator

aggregator.wrapper('/home/gcideron/visual_her/out/Fetch-12x12-N2-C6-SE5-SI5-O10-experts-to-learn/dueling-double-dqn-her', output="summary")

#import os

#path = '/home/gcideron/visual_her/out/Fetch-12x12-N2-C6-SE5-SI5-O10-experts-to-learn/dueling-double-dqn-her'

#for p in next(os.walk(path))[1]:
#    print(p)
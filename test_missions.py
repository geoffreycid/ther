import random

types = ["key", "ball"]
colors = ["rouge", "blue"]
seniority = ["old", "young"]
size = ["small", "average", "large"]


all_possible_missions = []
for c in colors:
    for t in types:
        for se in seniority:
            for si in size:
                mission = {
                    "color": c,
                    "type": t,
                    "seniority": se,
                    "size": si
                }
                all_possible_missions.append(mission)

random.shuffle(all_possible_missions)
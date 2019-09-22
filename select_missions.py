import random
miss_colors = ["red", "green", "blue", "purple", "yellow", "grey"]
miss_types = ["key", "ball"]
miss_sizes = ["tiny", "small", "medium", "large", "giant"]
miss_shades = ["very_light", "light", "neutral", "dark", "very_dark"]
all_missions = []
for color in miss_colors:
    for type in miss_types:
        for size in miss_sizes:
            for shade in miss_shades:
                all_missions.append([color, type, size, shade])

random.shuffle(all_missions)
train_missions = all_missions[:int(len(all_missions) * 0.1)]
hold_out_missions = all_missions[int(len(all_missions) * 0.1):]

a = []
for miss in train_missions:
    a += miss
print(len(set(a)))

a = []
for miss in hold_out_missions:
    a += miss
print(len(set(a)))

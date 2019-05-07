import tensorboardX as tb

writer = tb.SummaryWriter(log_dir="./summaries/")

for i in range(100):
    writer.add_scalar("data/reward", i, global_step=i+1)

writer.close()

print("over")

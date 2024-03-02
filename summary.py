import re
#from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

#writer = SummaryWriter("logs")

log_file = "./errncc.log"

losses = []

pattern = r"12663/12663"
pattern_total = r'total:\s*(-?[0-9.]+):'

with open(log_file, "r") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            mtch = re.search(pattern_total, line)
            if mtch:
                loss = float(mtch.group(1))
                losses.append(loss)
#writer.close()

print(losses)
print(len(losses))

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("NCC.png")
plt.show()

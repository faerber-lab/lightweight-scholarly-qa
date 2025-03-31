import torch
for i in range(torch.cuda.device_count()):
       print(torch.cuda.get_device_properties(i).name)

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(f"{available_gpus=}")

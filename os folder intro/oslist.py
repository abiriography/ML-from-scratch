import os
folders = os.listdir("data")
print(folders)


for folder in folders:
    print(folders)
    print(os.listdir(f"data/{folder}"))

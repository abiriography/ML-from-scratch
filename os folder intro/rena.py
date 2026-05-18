import os

if(not os.path.exists("data")):
    os.mkdir("data")

for i in range(0,100):
    os.rename(f"data/tutorial{i+1}", f"data/tutorial {i+1}")
#!/bin/bash

CUDA_VISIBLE_DEVICES=1 pm2 start --name "warm" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJwarm --netuid 26 --subtensor.network finney --axon.port 40045
CUDA_VISIBLE_DEVICES=2 pm2 start --name "hot" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJhot --netuid 26 --subtensor.network finney --axon.port 40065
# CUDA_VISIBLE_DEVICES=3 pm2 start --name "jja" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJa --netuid 26 --subtensor.network finney --axon.port 40175
CUDA_VISIBLE_DEVICES=4 pm2 start --name "jjb" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJb --netuid 26 --subtensor.network finney --axon.port 40217
CUDA_VISIBLE_DEVICES=5 pm2 start --name "jjc" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJc --netuid 26 --subtensor.network finney --axon.port 40238
CUDA_VISIBLE_DEVICES=6 pm2 start --name "jjd" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJd --netuid 26 --subtensor.network finney --axon.port 40292
CUDA_VISIBLE_DEVICES=7 pm2 start --name "jje" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJe --netuid 26 --subtensor.network finney --axon.port 40294
CUDA_VISIBLE_DEVICES=8 pm2 start --name "jjf" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJf --netuid 26 --subtensor.network finney --axon.port 40302
CUDA_VISIBLE_DEVICES=9 pm2 start --name "jjg" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJg --netuid 26 --subtensor.network finney --axon.port 40374
CUDA_VISIBLE_DEVICES=10 pm2 start --name "jjh" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJh --netuid 26 --subtensor.network finney --axon.port 40463
CUDA_VISIBLE_DEVICES=11 pm2 start --name "jji" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJi --netuid 26 --subtensor.network finney --axon.port 40606
CUDA_VISIBLE_DEVICES=12 pm2 start --name "jjj" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJj --netuid 26 --subtensor.network finney --axon.port 40672
CUDA_VISIBLE_DEVICES=14 pm2 start --name "jjl" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJl --netuid 26 --subtensor.network finney --axon.port 40745
CUDA_VISIBLE_DEVICES=15 pm2 start --name "jjm" python3 -- neurons/miners/StableMiner/main.py --wallet.name JJcold --wallet.hotkey JJm --netuid 26 --subtensor.network finney --axon.port 40787


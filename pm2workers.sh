#!/bin/bash

CUDA_VISIBLE_DEVICES=0 pm2 start python3 --name "worker0" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=1 pm2 start python3 --name "worker1" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=2 pm2 start python3 --name "worker2" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=3 pm2 start python3 --name "worker3" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=4 pm2 start python3 --name "worker4" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=5 pm2 start python3 --name "worker5" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=6 pm2 start python3 --name "worker6" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=7 pm2 start python3 --name "worker7" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=8 pm2 start python3 --name "worker8" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=9 pm2 start python3 --name "worker9" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=10 pm2 start python3 --name "worker10" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=11 pm2 start python3 --name "worker11" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=12 pm2 start python3 --name "worker12" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=13 pm2 start python3 --name "worker13" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=14 pm2 start python3 --name "worker14" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=15 pm2 start python3 --name "worker15" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=16 pm2 start python3 --name "worker16" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1
CUDA_VISIBLE_DEVICES=17 pm2 start python3 --name "worker17" -- /usr/local/bin/taskiq worker neurons.miners.StableMiner.Tasks:broker --workers 1


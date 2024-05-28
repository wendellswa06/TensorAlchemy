<div align="center">

![TensorAlchemy - Splash image](../../TensorAlchemy-splash.png)

# **MAINNET GUIDE FOR MINERS** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This documentation is for starting and running a miner on the ImageAlchemy mainnet.


## MINER REQUIREMENTS
- ⚗️ NVIDIA 4090 | A6000 or better
- ⚗️ Ubuntu 20.04 or 22.04
- ⚗️ Python 3.9 or 3.10
- ⚗️ CUDA 12.0 or higher


Below are several guides which include instructions to get mining with the Image Alchemy subnet.

## GUIDE 1 (venv)

### STEP 1. CLONE REPOSITORY
`git clone https://github.com/Supreme-Emperor-Wang/ImageAlchemy.git ~/ImageAlchemy`

### STEP 2. INSTALL PREREQUISITES
`sudo apt-get update && sudo apt-get install python3.10-venv`

### STEP 3. CREATE A VENV
`python3.10 -m venv ~/venvs/ImageAlchemy && source ~/venvs/ImageAlchemy/bin/activate && pip install wheel && pip install --upgrade setuptools`

### STEP 4. INSTALL REQUIREMENTS INTO VENV
`source ~/venvs/ImageAlchemy/bin/activate && cd ~/ImageAlchemy && pip install -e .`

### STEP 5. LAUNCH MINER
```bash
python ~/ImageAlchemy/neurons/miners/StableMiner/main.py --wallet.name NAME --wallet.hotkey HOTKEY --netuid 26 --subtensor.network finney --axon.port 8101 --miner.device cuda:0 --logging.debug --wandb.entity ENTITY --wandb.project PROJECT --wandb.api_key APIKEY
```

### NOTES

- ⚗️ You can remove the wandb options from the start command if you don’t want to log your miner’s images directly to your own wandb account.
- ⚗️ You can try adding the `--miner.optimize` flag to improve inference speed
PREREQUISITES FOR GUIDE 2 & 3

## INSTALL PM2

### INSTALL REQUIRED PACKAGES
```bash
sudo apt-get update && sudo apt-get install curl -y && curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash && source ~/.bashrc && nvm install node && npm install -g npm && npm install pm2 -g && curl -sL https://raw.githubusercontent.com/Unitech/pm2/master/packager/setup.deb.sh | sudo -E bash - && echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p && pm2 install pm2-logrotate && pm2 set pm2-logrotate:max_size 50M && pm2 startup && pm2 save && pm2 ls
```

### SET UP START UP SCRIPT
The final output may tell you "To set up the Startup Script, copy paste the following command..."
Copy paste it and hit enter. If you don’t see this message, you can move on.

### REMOVE THE KEYMETRICS APT REPOSITORY
`sudo rm /etc/apt/sources.list.d/Keymetrics_pm2.list`


## GUIDE 2 (pm2 + venv)


### STEP 1. CLONE REPOSITORY
`git clone https://github.com/Supreme-Emperor-Wang/ImageAlchemy.git ~/ImageAlchemy`


### STEP 2. INSTALL PREREQUISITES
`sudo apt-get update && sudo apt-get install python3.10-venv`


### STEP 3. CREATE A VENV
`python3.10 -m venv ~/venvs/ImageAlchemy && source ~/venvs/ImageAlchemy/bin/activate && pip install wheel && pip install --upgrade setuptools`

### STEP 4. INSTALL REQUIREMENTS INTO VENV
`source ~/venvs/ImageAlchemy/bin/activate && cd ~/ImageAlchemy && pip install -e .`

### STEP 5. LAUNCH MINER
```bash
pm2 start ~/ImageAlchemy/neurons/miners/StableMiner/main.py --interpreter ~/venvs/ImageAlchemy/bin/python --restart-delay 30000 --name NAME --  --wallet.name WALLET --wallet.hotkey HOTKEY --axon.port PORT --netuid 26 --subtensor.network finney --miner.device cuda:0 --logging.debug --wandb.entity ENTITY --wandb.project PROJECT --wandb.api_key APIKEY
```

### NOTES
- ⚗️ You can try adding the `--miner.optimize` flag to improve inference speed


## GUIDE 3 (pm2 + conda)

### STEP 1. CLONE REPOSITORY
`git clone https://github.com/Supreme-Emperor-Wang/ImageAlchemy.git ~/ImageAlchemy`


### STEP 2. INSTALL CONDA

### Change to root user directory and download Anaconda
`cd && wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh`

### Change permissions on the Anaconda installer
`chmod 777 ~/Anaconda3-2023.09-0-Linux-x86_64.sh`

### Install Anaconda
`./Anaconda3-2023.09-0-Linux-x86_64.sh`

Follow the prompts.

Enter. Accept agreement. Enter. Type yes.

#### Create conda environment
`conda create -n py310 python=3.10`

#### Activate conda env
`conda activate py310`

#### Prevent conda from activating automatically (optional):
`conda config --set auto_activate_base false`

### STEP 3. INSTALL REQUIREMENTS INTO CONDA ENV
`cd ~/ImageAlchemy && pip install -e .`

### STEP 4. LAUNCH MINER

```bash
pm2 start ~/ImageAlchemy/neurons/miners/StableMiner/main.py --interpreter ~/anaconda3/envs/py310/bin/python --restart-delay 30000 --name NAME --  --wallet.name WALLET --wallet.hotkey HOTKEY --axon.port PORT --netuid 26 --subtensor.network finney --miner.device cuda:0 --logging.debug --wandb.entity ENTITY --wandb.project PROJECT --wandb.api_key APIKEY
```

### NOTES
- ⚗️ You can try adding the `--miner.optimize` flag to improve inference speed

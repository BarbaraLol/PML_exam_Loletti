sudo apt-get remove --purge '^nvidia-.*' '^cuda-.*' -y
sudo apt-get autoremove -y
sudo apt-get autoclean
sudo rm -f /etc/apt/sources.list.d/cuda-*.list
sudo rm -f /etc/apt/preferences.d/cuda-*
sudo apt-get update
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update
sudo apt-get install nvidia-driver-550 -y

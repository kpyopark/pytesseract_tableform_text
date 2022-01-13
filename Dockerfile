FROM mylamour/tesseract-ocr:opencv
# https://linuxize.com/post/how-to-install-python-3-9-on-ubuntu-20-04/
apt update

apt install -y software-properties-common

apt update

echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main" >> /etc/apt/sources.list

apt-get update 2>&1 | grep NO_PUBKEY

# W: GPG error: http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY BA6932366A755776

apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776

apt-get update

apt install -y python3.8

apt install -y python3.8-distutils

rm /usr/bin/python3

ln -s /usr/bin/python3.8 /usr/bin/python3

cd /home

git clone https://github.com/kpyopark/pytesseract_tableform_text.git

cd /home/pytesseract_tableform_text

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

python3 get-pip.py

python3 -m pip install -r requirements.txt
# python3 -m pip install pytesseract

#FROM mylamour/tesseract-ocr:opencv
# https://linuxize.com/post/how-to-install-python-3-9-on-ubuntu-20-04/

FROM public.ecr.aws/lts/ubuntu:latest

COPY ./*.py /home/pytesseract_tableform_ocr/
COPY ./requirements.txt /home/pytesseract_tableform_ocr/

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt install -y tesseract-ocr python3.8 python3.8-distutils curl git libgl-dev
# RUN ln -s /usr/bin/python3.8 /usr/bin/python3

# RUN cd /home; git clone https://github.com/kpyopark/pytesseract_tableform_text.git
RUN curl https://bootstrap.pypa.io/get-pip.py -o /home/pytesseract_tableform_ocr/get-pip.py
RUN /usr/bin/python3.8 /home/pytesseract_tableform_ocr/get-pip.py
RUN cd /home/pytesseract_tableform_ocr; /usr/bin/python3.8 -m pip install -r requirements.txt

RUN mkdir /usr/local/share/tessdata
RUN curl https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/kor.traineddata -o /usr/local/share/tessdata/kor.traineddata

CMD [ "sh", "-c", "export TESSDATA_PREFIX=/usr/local/share/tessdata;/usr/bin/python3.8 -u /home/pytesseract_tableform_ocr/sample.py STDIN" ]

# apt install -y software-properties-common
# apt update
# echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main" >> /etc/apt/sources.list
# apt-get update 2>&1 | grep NO_PUBKEY
# W: GPG error: http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY BA6932366A755776
# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
# apt-get update
# apt install -y python3.8
# rm /usr/bin/python3
# ln -s /usr/bin/python3.8 /usr/bin/python3










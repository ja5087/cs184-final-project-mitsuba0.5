FROM ubuntu:focal

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install qttools5-dev-tools libqt5xmlpatterns5-dev qt5-default qtcreator libglewmx-dev python3-pip
RUN pip3 install scons

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install build-essential libpng-dev libjpeg-dev libilmbase-dev libxerces-c-dev libboost-all-dev libopenexr-dev libglewmx-dev libxxf86vm-dev libpcrecpp0v5 libeigen3-dev libfftw3-dev


# Muon denoising project


-------------------------------------------------------------
1.- Install GEANT4 from source and compile
-------------------------------------------------------------

1.1 Install pre-requisites

+ Make sure to have at least GCC11 or greater.

+ CMake 3.16 or higher.

In any case, check: 

https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/gettingstarted.html

+ Pre-requesites (If you are running on your own local computer)

sudo apt-get -y install build-essential openssl libssl-dev libssl1.0 libgl1-mesa-dev libqt5x11extras5

sudo apt-get install qtbase5-dev

sudo apt-get install qtdeclarative5-dev

sudo apt-get install -y libxmu-dev


1.2 Get the GEANT4 source code 

Get the latest release from the GEANT4 page. Version: 4.11.2.1.

wget https://gitlab.cern.ch/geant4/geant4/-/archive/v11.2.1/geant4-v11.2.1.tar.gz

1.3 Setup installation and compile

gunzip geant4-v11.2.1.tar.gz

tar -xvf geant4-v11.2.1.tar

rm geant4-v11.2.1.tar

mkdir geant4-v11.2.1-build

cd geant4-v11.2.1-build

cmake -DCMAKE_INSTALL_PREFIX=../geant4-v11.1.2-install -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_SYSTEM_EXPAT=OFF -DGeant4_USE_OPENGL_X11=ON -DGeant4_USE_QT=ON ../geant4-v11.2.1

make -j8

make install

1.4 Add information to setup.sh

Write the path to the install version of geant4 in setup.sh

-------------------------------------------------------------
2.- Install ROOT from source and compile
-------------------------------------------------------------

2.1 Install pre-requisites (Only if you are running on your own computer)

In general cmake will ask for dependencies, but some of the most normal missing dependencies can be installed with:

sudo apt-get install libxpm-dev libxpm4 libxft-dev libxft2 libxext-dev libxext6


2.2 Get the ROOT source code

wget https://root.cern/download/root_v6.28.04.source.tar.gz


2.3 Setup installation and compile 

gunzip root_v6.28.04.source.tar.gz

tar -xvf root_v6.28.04.source.tar

rm root_v6.28.04.source.tar

mkdir root_v6.28.04-build root_v6.28.04-install

cd root_v6.28.04-build

cmake -DCMAKE_INSTALL_PREFIX=../root_v6.28.04-install ../root-6.28.04

cmake --build . --target install

echo "source $HOME/root_v6.28.04-install/bin/thisroot.sh" >> $HOME/.bashrc


-------------------------------------------------------------
3.- Install JSON library
-------------------------------------------------------------

You can do this by just copy-pasting these lines:

#Make sure you are outside of geant4 and root release areas

git clone https://github.com/parbol/jsoncpp.git

mkdir jsoncpp-build

cd jsoncpp-build

cmake -DJSONCPP_WITH_PKGCONFIG_SUPPORT=OFF -DJSONCPP_WITH_TESTS=OFF ../jsoncpp/

make

cd $HOME/

mkdir jsoncpp/lib/

cp jsoncpp-build/lib/libjsoncpp.a jsoncpp/lib/

-------------------------------------------------------------
4.- Install Muography_Denoising
-------------------------------------------------------------

4.1 Get Muography_Denoising

Muography_Denoising can be obtained from github:

git clone https://github.com/parbol/Muography_Denoising.git

Adapt the setup.sh file to point to the path in which you have installed Muography_Denoising.

source setup.sh

-------------------------------------------------------------
5.- Setup CRY generator
-------------------------------------------------------------

cd cry/cry_v1.7/

source setup.sh

make


-------------------------------------------------------------
6.- Setup MuonGeneration software (telescope simulator)
-------------------------------------------------------------

6.1 Setup the Makefile

This step only needs to be done once and it will produce the Makefile for the project. Unless
there are changes in the libraries this process should not be repeated.

cd $G4WORKDIR

source setup.sh

mkdir MuonGeneration-build

cd MuonGeneration-build

cmake -DGeant4_DIR=$G4INSTALLDIR ../MuonGeneration/

6.2 Compile program

In MuonGeneration-build simply type "make"



//----------------------------------------------------------------------//
// ___  ___                    _____           _                        //
// |  \/  |                   /  ___|         | |                       //
// | .  . |_   _  ___  _ __   \ `--. _   _ ___| |_ ___ _ __ ___  ___    //
// | |\/| | | | |/ _ \| '_ \   `--. \ | | / __| __/ _ \ '_ ` _ \/ __|   //
// | |  | | |_| | (_) | | | | /\__/ / |_| \__ \ ||  __/ | | | | \__ \   //
// \_|  |_/\__,_|\___/|_| |_| \____/ \__, |___/\__\___|_| |_| |_|___/   //
//                                    __/ |                             //
//----------------------------------------------------------------------//
// A project by: C. Diez, P. Gomez and P. Martinez                      //
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//
// PrimaryGenerationAction.cc                                           //
//----------------------------------------------------------------------//
// This class handles the generation of events by calling CRY.          //
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

#include <iomanip>
#include "PrimaryGeneratorAction.hh"
#include <sstream>
#include <sys/time.h>

#include "G4Event.hh"

using namespace std;



//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
PrimaryGeneratorAction::PrimaryGeneratorAction(ConfigurationGeometry *myGeom_, G4String inputfile, G4long randomSeed_, G4double pt_) {

    myGeom = myGeom_;
    randomSeed = randomSeed_;
    pt = pt_;

    particleGun = new G4ParticleGun();
    timeSimulated=0.0;

    CRYFromFile(inputfile);

    // create a vector to store the CRY particle properties
    vect=new std::vector<CRYParticle*>;

    // Create the table containing all particle names
    particleTable = G4ParticleTable::GetParticleTable();

    // Create the messenger file
    gunMessenger = new PrimaryGeneratorMessenger(this);
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Destructuor                                                          //
//----------------------------------------------------------------------//
PrimaryGeneratorAction::~PrimaryGeneratorAction() {

    G4cout << "Time simulated: " << gen->timeSimulated() << G4endl;

}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Don't know very well why we need this                                //
//----------------------------------------------------------------------//
void PrimaryGeneratorAction::InputCRY() {
    InputState=1;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Update CRY state                                                     //
//----------------------------------------------------------------------//
void PrimaryGeneratorAction::UpdateCRY(std::string* MessInput) {

    CRYSetup *setup=new CRYSetup(*MessInput,".");

    gen = new CRYGenerator(setup);

    // set random number generator
    long seedValue;

    if(randomSeed != 0) {
        seedValue = (long) randomSeed;
    } else {
        timeval tim;
        gettimeofday(&tim, NULL);
        double t1=tim.tv_sec+(tim.tv_usec)*100000;
        if(t1<0) t1=t1*(-1);
        seedValue=(long) t1;
    }

    CLHEP::HepRandomEngine* MyRndEngine = CLHEP::HepRandom::getTheEngine();
    MyRndEngine->setSeed(seedValue,1);
    RNGWrapper<CLHEP::HepRandomEngine>::set(MyRndEngine,&CLHEP::HepRandomEngine::flat);
    setup->setRandomFunction(RNGWrapper<CLHEP::HepRandomEngine>::rng);

    InputState=0;

}


//----------------------------------------------------------------------//
// Update CRY state                                                     //
//----------------------------------------------------------------------//
void PrimaryGeneratorAction::CRYFromFile(G4String newValue) {

    std::ifstream inputFile;
    inputFile.open(newValue,std::ios::in);
    char buffer[1000];

    std::string setupString("");
    std::ostringstream os;
    os << (myGeom->getSizeBoxCRY()/CLHEP::cm)/100.0;
    std::string theSize = os.str();
    if (inputFile.fail()) {
        setupString.append("returnNeutrons 0");
        setupString.append(" ");
        setupString.append("returnProtons 0");
        setupString.append(" ");
        setupString.append("returnGammas 0");
        setupString.append(" ");
        setupString.append("date 7-1-2012");
        setupString.append(" ");
        setupString.append("latitude 90.0");
        setupString.append(" ");
        setupString.append("altitude 0");
        setupString.append(" ");
        setupString.append("subboxLength " + theSize);
        setupString.append(" ");
    } else {

        while ( !inputFile.getline(buffer,1000).eof()) {
            setupString.append(buffer);
            setupString.append(" ");
        }

    }

    CRYSetup *setup=new CRYSetup(setupString, ".");

    gen = new CRYGenerator(setup);

    long seedValue;
    if(randomSeed != 0) {
        seedValue = (long) randomSeed;
    } else {
        timeval tim;
        gettimeofday(&tim, NULL);
        double t1=tim.tv_sec+(tim.tv_usec)*100000;
        if(t1<0) t1=t1*(-1);
        seedValue=(long) t1;
    }
    // set random number generator
    CLHEP::HepRandomEngine* MyRndEngine = CLHEP::HepRandom::getTheEngine();
    MyRndEngine->setSeed(randomSeed,1);
    RNGWrapper<CLHEP::HepRandomEngine>::set(MyRndEngine,&CLHEP::HepRandomEngine::flat);
    setup->setRandomFunction(RNGWrapper<CLHEP::HepRandomEngine>::rng);

    InputState=0;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Real generation of the primaries                                     //
//----------------------------------------------------------------------//
void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent) {


    if (InputState != 0) {
        G4String* str = new G4String("CRY library was not successfully initialized");
        G4Exception("PrimaryGeneratorAction", "1", RunMustBeAborted, *str);
    }
    G4String particleName;
    vect->clear();
    gen->genEvent(vect);


    for ( unsigned j=0; j<vect->size(); j++) {
        particleName=CRYUtils::partName((*vect)[j]->id());
        particleGun->SetParticleDefinition(particleTable->FindParticle((*vect)[j]->PDGid()));
        if(pt == 0) {
		    particleGun->SetParticleEnergy((*vect)[j]->ke()*CLHEP::MeV);
		} else {
            particleGun->SetParticleEnergy(pt*CLHEP::MeV);
		}
        particleGun->SetParticlePosition(G4ThreeVector((*vect)[j]->x()*CLHEP::m, (*vect)[j]->y()*CLHEP::m, ((*vect)[j]->z()+(myGeom->getZOffsetCRY()/CLHEP::cm)/100.0)*CLHEP::m));
        particleGun->SetParticleMomentumDirection(G4ThreeVector((*vect)[j]->u(), (*vect)[j]->v(), (*vect)[j]->w()));
        particleGun->SetParticleTime((*vect)[j]->t());
        particleGun->GeneratePrimaryVertex(anEvent);
        //G4cout << particleName << G4endl;
        //G4cout << "Position " << ((*vect)[j]->x()*CLHEP::m)/CLHEP::cm << " " << ((*vect)[j]->y()*CLHEP::m)/CLHEP::cm << " " <<  (((*vect)[j]->z()+(myGeom->getZOffsetCRY()/CLHEP::cm)/100.0)*CLHEP::m)/CLHEP::cm << G4endl;
        //G4cout << "Energy "   << (*vect)[j]->ke() << G4endl;
        //G4cout << "Momentum " << (*vect)[j]->u() << " " << (*vect)[j]->v() << " " <<  (*vect)[j]->w() << G4endl;
        delete (*vect)[j];
    }
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


#include "LayerSensorHit.hh"
#include "G4ios.hh"
#include "G4VVisManager.hh"
#include "G4Circle.hh"
#include "G4Colour.hh"
#include "G4AttDefStore.hh"
#include "G4AttDef.hh"
#include "G4AttValue.hh"
#include "G4UIcommand.hh"
#include "G4UnitsTable.hh"
#include "G4VisAttributes.hh"
#include "G4LogicalVolume.hh"

//----------------------------------------------------------------------//
// Declaration of the allocator                                         //
//----------------------------------------------------------------------//
G4Allocator<LayerSensorHit> LayerSensorHitAllocator;
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Constructor without indicating layer ID                              //
//----------------------------------------------------------------------//
LayerSensorHit::LayerSensorHit() {
    ;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Destructor                                                           //
//----------------------------------------------------------------------//
LayerSensorHit::~LayerSensorHit() {
    ;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Constructor from another hit                                         //
//----------------------------------------------------------------------//
LayerSensorHit::LayerSensorHit(const LayerSensorHit &right): G4VHit() {
    eventNumber = right.eventNumber;
    detectorID = right.detectorID;
    layerID = right.layerID;
    energy = right.energy;
    localPos = right.localPos;
    globalPos = right.globalPos;
    localDir = right.localDir;
    globalDir = right.globalDir;
    genID = right.genID;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Overloading = operator                                               //
//----------------------------------------------------------------------//
const LayerSensorHit& LayerSensorHit::operator=(const LayerSensorHit &right) {
    eventNumber = right.eventNumber;
    detectorID = right.detectorID;
    layerID = right.layerID;
    energy = right.energy;
    localPos = right.localPos;
    globalPos = right.globalPos;
    localDir = right.localDir;
    globalDir = right.globalDir;
    genID = right.genID;
    return *this;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Overloading == operator                                               //
//----------------------------------------------------------------------//
int LayerSensorHit::operator==(const LayerSensorHit &/*right*/) const {
    return 0;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Associate measurements and units                                     //
//----------------------------------------------------------------------//
const std::map<G4String,G4AttDef>* LayerSensorHit::GetAttDefs() const {
    G4bool isNew;
    std::map<G4String,G4AttDef>* store = G4AttDefStore::GetInstance("LGADSensorHit",isNew);
    if (isNew) {
        G4String ID("ID");
        (*store)[ID] = G4AttDef(ID,"ID","Physics","","G4int");

        G4String Time("Time");
        (*store)[Time] = G4AttDef(Time,"Time","Physics","G4BestUnit","G4double");

        G4String Energy("Energy");
        (*store)[Energy] = G4AttDef(Energy,"Energy","Physics","G4BestUnit","G4double");

        G4String Pos("Pos");
        (*store)[Pos] = G4AttDef(Pos, "Position", "Physics","G4BestUnit","G4ThreeVector");

    }
    return store;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Create specials fuields inside the hit                               //
//----------------------------------------------------------------------//
std::vector<G4AttValue>* LayerSensorHit::CreateAttValues() const {
    std::vector<G4AttValue>* values = new std::vector<G4AttValue>;

    values->push_back(G4AttValue("HitType","LGADSensorHit",""));

    values->push_back
    (G4AttValue("ID",G4UIcommand::ConvertToString(layerID),""));

    values->push_back
    (G4AttValue("Energy",G4BestUnit(energy,"Energy"),""));

    values->push_back
    (G4AttValue("Pos",G4BestUnit(globalPos,"Length"),""));

    return values;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Printing the information of the hit                                  //
//----------------------------------------------------------------------//
void LayerSensorHit::Print()
{
    G4cout << "\033[1;34m"
           << "*******************************Hit**********************************" << G4endl
           << "Event number: " << "\033[1;33m" << eventNumber << "\033[1;34m" << G4endl
           << "Detector: " << "\033[1;33m" << detectorID << "\033[1;34m" << G4endl
           << "Layer: " << "\033[1;33m" << layerID << "\033[1;34m" << G4endl
           << "Energy: " << "\033[1;33m" << energy/CLHEP::GeV << "\033[1;34m" << G4endl
           << "Local pos: " << "\033[1;33m" << localPos.x()/CLHEP::cm << " " << localPos.y()/CLHEP::cm << " " << localPos.z()/CLHEP::cm << "\033[1;34m" << G4endl
           << "Local pos: " << "\033[1;33m" << globalPos.x()/CLHEP::cm << " " << globalPos.y()/CLHEP::cm << " " << globalPos.z()/CLHEP::cm << "\033[1;34m" << G4endl
           << "********************************************************************" << "\033[0m" << G4endl;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//



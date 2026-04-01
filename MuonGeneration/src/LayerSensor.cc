#include "LayerSensor.hh"
#include "LayerSensorHit.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4EventManager.hh"
#include "G4Navigator.hh"
#include "G4ios.hh"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"

#include <tuple>

//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
LayerSensor::LayerSensor(G4String name, G4String collection)
    :G4VSensitiveDetector(name)
{
    G4String HCname;
    collectionName.insert(collection);
    HCID = -1;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
LayerSensor::~LayerSensor() {
    ;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
void LayerSensor::Initialize(G4HCofThisEvent*HCE)
{
    hitsCollection = new LayerSensorHitsCollection(SensitiveDetectorName, collectionName[0]);
    if(HCID<0) HCID = G4SDManager::GetSDMpointer()->GetCollectionID(hitsCollection);
    HCE->AddHitsCollection(HCID,hitsCollection);
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Process the hits and fills the relevant information                  //
//----------------------------------------------------------------------//
G4bool LayerSensor::ProcessHits(G4Step*aStep,G4TouchableHistory*  /*ROhist*/) {

    G4Event *event = G4EventManager::GetEventManager()->GetEventManager()->GetNonconstCurrentEvent();
    if(event == NULL) false;
    G4int enumber = event->GetEventID();
    G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
    G4TouchableHistory* theTouchable = (G4TouchableHistory*)(preStepPoint->GetTouchable());
  
    G4ThreeVector worldPos = preStepPoint->GetPosition();
    G4ThreeVector localPos = theTouchable->GetHistory()->GetTopTransform().TransformPoint(worldPos);
    
    G4ThreeVector worldDir = preStepPoint->GetMomentumDirection();
    G4ThreeVector localDir = theTouchable->GetHistory()->GetTopTransform().TransformAxis(worldDir);
    
    G4int detector = layer->detId();
    G4int layer_ = layer->layerId();  
    G4double energy = aStep->GetPreStepPoint()->GetTotalEnergy();
    G4int genID = aStep->GetTrack()->GetParticleDefinition()->GetPDGEncoding();

    //Simulating resolution
    LayerSensorHit* aHit = new LayerSensorHit();
    aHit->SetEventNumber(enumber);
    aHit->SetDetectorID(detector);
    aHit->SetLayerID(layer_);
    aHit->SetLocalPos(localPos);
    aHit->SetGlobalPos(worldPos);
    aHit->SetLocalDir(localDir);
    aHit->SetGlobalDir(worldDir);
    aHit->SetEnergy(energy);
    aHit->SetGenID(genID);
    
    hitsCollection->insert(aHit);

    return true;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Set layer                                                            //
//----------------------------------------------------------------------//
void LayerSensor::setLayer(Layer *a) {
    layer = a;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Get lgad                                                             //
//----------------------------------------------------------------------//
Layer * LayerSensor::getLayer() {
    return layer;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Needed by the parent class                                           //
//----------------------------------------------------------------------//
void LayerSensor::EndOfEvent(G4HCofThisEvent* /*HCE*/) {
    ;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


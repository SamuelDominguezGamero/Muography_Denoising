#include "EventAction.hh"
#include "EventActionMessenger.hh"

#include "G4AnalysisManager.hh"

#include "Analysis.hh"
#include "G4Run.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"
#include "G4TrajectoryContainer.hh"
#include "G4Trajectory.hh"
#include "G4VVisManager.hh"
#include "G4SDManager.hh"
#include "G4UImanager.hh"
#include "G4ios.hh"

#include "LayerSensorHit.hh"

#include "ConfigurationGeometry.hh"

#include <math.h>

#include <iostream>
#include <map>

#define PI 3.141592653589793238462643383279502884



//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
EventAction::EventAction(ConfigurationGeometry *geom_) {

    geom = geom_;
    G4String colName;
    G4SDManager* SDman = G4SDManager::GetSDMpointer();
    G4int numberOfDetectors = SDman->GetCollectionCapacity();
    SDman->ListTree();

    for(auto i : geom->collections) {
        DHCID.push_back(SDman->GetCollectionID(i));
    }
    messenger = new EventActionMessenger(this);

}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Destructor                                                           //
//----------------------------------------------------------------------//
EventAction::~EventAction() {
    delete messenger;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//



//----------------------------------------------------------------------//
// Begin of the event                                                   //
//----------------------------------------------------------------------//
void EventAction::BeginOfEventAction(const G4Event*) {}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// End of the event -> the action goes here                             //
//----------------------------------------------------------------------//
void EventAction::EndOfEventAction(const G4Event* evt) {

    auto man = G4AnalysisManager::Instance();
    G4HCofThisEvent * HCE = evt->GetHCofThisEvent();
    
    std::vector<LayerSensorHitsCollections> DHCs;
    //G4SDManager* SDman = G4SDManager::GetSDMpointer();
    if(HCE) {
        for (auto i : DHCID) {
            LayerSensorHitsCollection *a = 0;
            a = (LayerSensorHitsCollection*)(HCE->GetHC(i));
            LayerSensorHitsCollections DHCcoll;
            DHCcoll.push_back(a);
            DHCs.push_back(DHCcoll);
        }
        for (auto i : DHCs) {
            if(i.at(0)) {
                G4int n_hit = i.at(0)->entries();
                for(G4int hit = 0; hit < n_hit; hit++) {
                    LayerSensorHit* aHit = (*(i.at(0)))[hit];
                    if(aHit->GetEnergy() == 0) continue;
                    auto detID = aHit->GetDetectorID();
                    auto layerID = aHit->GetLayerID();
                    man->FillNtupleIColumn(0, aHit->GetEventNumber());
                    man->FillNtupleIColumn(1, aHit->GetDetectorID());
                    man->FillNtupleIColumn(2, aHit->GetLayerID());
                    man->FillNtupleDColumn(3, aHit->GetEnergy()/CLHEP::GeV);
                    man->FillNtupleDColumn(4, aHit->GetLocalPos().x()/CLHEP::cm);
                    man->FillNtupleDColumn(5, aHit->GetLocalPos().y()/CLHEP::cm);
                    man->FillNtupleDColumn(6, aHit->GetLocalPos().z()/CLHEP::cm);
                    man->FillNtupleDColumn(7, aHit->GetLocalDir().x());
                    man->FillNtupleDColumn(8, aHit->GetLocalDir().y());
                    man->FillNtupleDColumn(9, aHit->GetLocalDir().z());
                    man->FillNtupleDColumn(10, aHit->GetGlobalPos().x()/CLHEP::cm);
                    man->FillNtupleDColumn(11, aHit->GetGlobalPos().y()/CLHEP::cm);
                    man->FillNtupleDColumn(12, aHit->GetGlobalPos().z()/CLHEP::cm);
                    man->FillNtupleDColumn(13, aHit->GetGlobalDir().x());
                    man->FillNtupleDColumn(14, aHit->GetGlobalDir().y());
                    man->FillNtupleDColumn(15, aHit->GetGlobalDir().z());
                    man->FillNtupleIColumn(16, aHit->GetGenID());
                    man->AddNtupleRow();   
                }                    
            }
        }
    }
    
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

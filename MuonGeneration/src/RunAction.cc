#include "RunAction.hh"
#include "G4AnalysisManager.hh"

#include "Analysis.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"



//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
RunAction::RunAction(std::string name) {
    nameOfOutputFile = name;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Destructor                                                           //
//----------------------------------------------------------------------//
RunAction::~RunAction() {}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// We set up the columns when the run starts                            //
//----------------------------------------------------------------------//
void RunAction::BeginOfRunAction(const G4Run* run) {

    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->OpenFile(nameOfOutputFile.c_str());
    // Create ntuple
    man->CreateNtuple("hits", "The hits");
    
    man->CreateNtupleIColumn("eventNumber");
    man->CreateNtupleIColumn("det");
    man->CreateNtupleIColumn("layer");
    man->CreateNtupleDColumn("energy");
    man->CreateNtupleDColumn("localx");
    man->CreateNtupleDColumn("localy");
    man->CreateNtupleDColumn("localz");
    man->CreateNtupleDColumn("localvx");
    man->CreateNtupleDColumn("localvy");
    man->CreateNtupleDColumn("localvz");
    man->CreateNtupleDColumn("x");
    man->CreateNtupleDColumn("y");
    man->CreateNtupleDColumn("z");
    man->CreateNtupleDColumn("vx");
    man->CreateNtupleDColumn("vy");
    man->CreateNtupleDColumn("vz");
    man->CreateNtupleIColumn("genID");

    man->FinishNtuple();
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// At the end we store                                                  //
//----------------------------------------------------------------------//
void RunAction::EndOfRunAction(const G4Run* aRun)
{
    G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();
    analysisManager->Write();
    analysisManager->CloseFile();
    //delete G4AnalysisManager::Instance();
    
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


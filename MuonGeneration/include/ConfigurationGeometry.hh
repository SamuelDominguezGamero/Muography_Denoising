//------------------------------------------------------------//
// |__   __/ __ \|  \/  | |  | | |    / ____|   /\   |  __ \  //
//    | | | |  | | \  / | |  | | |   | |  __   /  \  | |  | | //
//    | | | |  | | |\/| | |  | | |   | | |_ | / /\ \ | |  | | //
//    | | | |__| | |  | | |__| | |___| |__| |/ ____ \| |__| | //
//    |_|  \____/|_|  |_|\____/|______\_____/_/    \_\_____/  //
//------------------------------------------------------------//
// ConfigurationGeometry class:                               //                                                           
//                                                            //
// Parses json files with the configuration of the detectors. //
//                                                            //
//------------------------------------------------------------//

#ifndef ConfigurationGeometry_h
#define ConfigurationGeometry_h 1

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "assert.h"

#include "globals.hh"
#include "Detector.hh"
#include "Layer.hh"
#include "Voxel.hh"

class ConfigurationGeometry {

public:

    ConfigurationGeometry(G4String);
    
    bool isGood();

    // Information about the GEANT4 universe 
    G4double getSizeX();
    G4double getSizeY();
    G4double getSizeZ();
   
    // Detector information
    Detector *getDetector(G4int);
    G4int getNDetectors();

    // Voxel information.
    // Voxels are passive cubic volumes placed in the world between detectors.
    // They do not register hits; they scatter/absorb particles according to
    // the physics list. Defined via VoxelConfig + TheVoxels in the JSON.
    Voxel *getVoxel(G4int);   // access voxel by 0-based index
    G4int getNVoxels();        // total number of voxels (0 if none in JSON)

    G4double getZOffsetCRY();
    G4double getSizeBoxCRY();

    //Creating the geometry
    void createG4objects(G4LogicalVolume *, 
                         std::map<G4String, G4Material*> &,
                         G4SDManager *);
    // Detector collections
    void registerCollection(G4String);
    std::vector <G4String> collections;
    
    void Print();

private:
    G4double uniSizeX, uniSizeY, uniSizeZ;
    G4double zOffsetCRY, sizeBoxCRY;
    G4double minPhi, maxPhi, minTheta, maxTheta;
    std::vector <Detector *> detectors; // list of detectors parsed from "Detectors" JSON array
    std::vector <Voxel *>    voxels;    // list of passive voxels parsed from "TheVoxels" JSON array (may be empty)
    bool goodGeometry;
    
};



#endif


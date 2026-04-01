#include "ConfigurationGeometry.hh"
#include <json/json.h>
#include <json/value.h>



//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
ConfigurationGeometry::ConfigurationGeometry(G4String file) {

    Json::Value root;
    Json::Reader reader;

    //We open the JSON file, read it and put it on a string
    std::stringstream  filecontent;
    std::ifstream infile(file.c_str());

    if(!(infile.good())) {
        G4cerr << "\033[1;31m" << "Error opening geometry file: " + file << "\033[0m" << G4endl;
        goodGeometry = false;
        return;
    }

    std::string currline;
    while(getline(infile,currline)) {
        filecontent<<currline;
    }

    infile.close();
    

    G4cout << "DEBUG: Lectura de archivo completada. Iniciando parser..." << G4endl;

    std::string FileContent=filecontent.str();

    //Parsing of the JSON file
    bool parsingSuccessful = reader.parse( FileContent, root );
    if ( !parsingSuccessful ) {
        G4cerr << "\033[1;31m" << "Error parsing file: " + file << "\033[0m" << G4endl;
        goodGeometry = false;
        return;
    }

    G4cout << "DEBUG: Parser exitoso. Accediendo a los datos..." << G4endl;


    if( root.size() > 0 ) {
        //Definition of the Universe ----------------------------------------------
        uniSizeX = atof(root["theWorld"]["xSizeWorld"].asString().c_str())*CLHEP::cm;
        uniSizeY = atof(root["theWorld"]["ySizeWorld"].asString().c_str())*CLHEP::cm;
        uniSizeZ = atof(root["theWorld"]["zSizeWorld"].asString().c_str())*CLHEP::cm;
        sizeBoxCRY = atof(root["theWorld"]["sizeBoxCRY"].asString().c_str())*CLHEP::cm;
	zOffsetCRY = atof(root["theWorld"]["zOffsetCRY"].asString().c_str())*CLHEP::cm;

        if(uniSizeX <= 0 || uniSizeY <= 0|| uniSizeZ <= 0) {
            G4cerr << "\033[1;31m" << "The size of the Universe has been smaller than 0" << "\033[0m" << G4endl;
            goodGeometry = false;
            return;
        }

        //Definition of the Detectors ----------------------------------------------
        const Json::Value Detectors = root["Detectors"];

	    for(G4int idet = 0; idet < Detectors.size(); ++idet) {

	        G4double xPos = atof(root["Detectors"][idet]["xPosDetector"].asString().c_str()) * CLHEP::cm;
            G4double yPos = atof(root["Detectors"][idet]["yPosDetector"].asString().c_str()) * CLHEP::cm;
            G4double zPos = atof(root["Detectors"][idet]["zPosDetector"].asString().c_str()) * CLHEP::cm;
            G4double xDir = atof(root["Detectors"][idet]["xDirDetector"].asString().c_str()) * CLHEP::degree;
            G4double yDir = atof(root["Detectors"][idet]["yDirDetector"].asString().c_str()) * CLHEP::degree;
            G4double zDir = atof(root["Detectors"][idet]["zDirDetector"].asString().c_str()) * CLHEP::degree;
            G4double xSize = atof(root["Detectors"][idet]["xSizeDetector"].asString().c_str()) * CLHEP::cm;
            G4double ySize = atof(root["Detectors"][idet]["ySizeDetector"].asString().c_str()) * CLHEP::cm;
            G4double zSize = atof(root["Detectors"][idet]["zSizeDetector"].asString().c_str()) * CLHEP::cm;
 
            if(xPos - xSize/2.0 < - uniSizeX/2.0 || yPos - ySize/2.0 < -uniSizeY/2.0 || zPos - zSize/2.0 < -uniSizeZ/2.0) {
                G4cerr << "\033[1;31m" << "Chamber " << idet << " is not contained in the world" << "\033[0m" << G4endl;
                goodGeometry = false;
                return;
            }
            Detector *detector = new Detector(xPos, yPos, zPos, xDir, yDir, zDir, xSize, ySize, zSize, idet);
            
            //Layers inside a detector ----------------------------------------------
	        const Json::Value jLayer = root["Detectors"][idet]["Layers"];
            
            for(G4int icoll = 0; icoll < jLayer.size(); ++icoll) {
                G4double xPosLayer = atof(root["Detectors"][idet]["Layers"][icoll]["xPosLayer"].asString().c_str()) * CLHEP::cm;
                G4double yPosLayer = atof(root["Detectors"][idet]["Layers"][icoll]["yPosLayer"].asString().c_str()) * CLHEP::cm;
                G4double zPosLayer = atof(root["Detectors"][idet]["Layers"][icoll]["zPosLayer"].asString().c_str()) * CLHEP::cm;
                G4double xDirLayer = atof(root["Detectors"][idet]["Layers"][icoll]["xDirLayer"].asString().c_str()) * CLHEP::degree;
                G4double yDirLayer = atof(root["Detectors"][idet]["Layers"][icoll]["yDirLayer"].asString().c_str()) * CLHEP::degree;
                G4double zDirLayer = atof(root["Detectors"][idet]["Layers"][icoll]["zDirLayer"].asString().c_str()) * CLHEP::degree;
                G4double xSizeLayer_ = atof(root["Detectors"][idet]["Layers"][icoll]["xSizeLayer"].asString().c_str()) * CLHEP::cm;
                G4double ySizeLayer_ = atof(root["Detectors"][idet]["Layers"][icoll]["ySizeLayer"].asString().c_str()) * CLHEP::cm;
                G4double zSizeLayer_ = atof(root["Detectors"][idet]["Layers"][icoll]["zSizeLayer"].asString().c_str()) * CLHEP::cm;
		        Layer *layer = new Layer(xPosLayer, yPosLayer, zPosLayer, xDirLayer, yDirLayer, zDirLayer, xSizeLayer_, ySizeLayer_, zSizeLayer_, idet, icoll);
                G4String label = G4String(std::to_string(idet)) + G4String("_") + 
                                     G4String(std::to_string(icoll)); 
                G4String Coll = G4String("HitsCollection_") + label;
                registerCollection(Coll); 
                detector->AddLayer(layer);
	        }
            detectors.push_back(detector);	    
        }
       
        G4cout << "DEBUG: Mundo y detectores cargados correctamente." << G4endl;
        
        //Definition of the Voxels -----------------------------------------------
        // The voxel block is optional: if the JSON has no "VoxelConfig" or
        // "TheVoxels" keys the simulation runs normally with zero voxels,
        // keeping full backward compatibility with existing config files.
        if(root.isMember("VoxelConfig") && root.isMember("TheVoxels")) {

            // --- Shared voxel configuration ---
            // These three values apply to EVERY voxel in TheVoxels array.
            // VoxelSize : side length of the cubic voxel in cm.
            //             Multiplied by CLHEP::cm to convert to Geant4 internal units (mm).
            // zPosVoxel : z coordinate (world frame) where all voxels are placed.
            //             All voxels share the same z, they differ only in x and y.
            // voxelMat  : string key used to look up a G4Material* in the materials map
            //             (e.g. "lead" maps to G4_Pb in DetectorConstruction).
            
            // ESTO ERA DE UNA CONFIGURACIÓN ANTIGUA, AHORA CADA VOXEL TIENE SU TAMAÑO Y MATERIAL DEFINIDO EN EL JSON, ASÍ QUE ESTO SE IGNORA_
            // ---------
            // G4double voxelSize = atof(root["VoxelConfig"]["VoxelSize"].asString().c_str()) * CLHEP::cm;
            // G4double zPosVoxel = atof(root["VoxelConfig"]["zPosVoxel"].asString().c_str()) * CLHEP::cm;
            // G4String voxelMat  = root["VoxelConfig"]["material"].asString();
            // ---------
            // Retrieve the array of voxel entries from the JSON.
            // Each entry contains only xPosVoxel and yPosVoxel; z comes from VoxelConfig.
            const Json::Value jVoxels = root["TheVoxels"];

            // Iterate over every entry in TheVoxels and build one Voxel object each.
            // ivox is used as the unique voxel ID (0-based sequential index).
            for(G4int ivox = 0; ivox < (G4int)jVoxels.size(); ++ivox) {

                // Read x, y, zpositions for this voxel (in cm → convert to CLHEP mm).
                G4double xPos = atof(jVoxels[ivox]["xPosVoxel"].asString().c_str()) * CLHEP::cm;
                G4double yPos = atof(jVoxels[ivox]["yPosVoxel"].asString().c_str()) * CLHEP::cm;
                G4double zPos = atof(jVoxels[ivox]["zPosVoxel"].asString().c_str()) * CLHEP::cm;
                G4double xSize = atof(jVoxels[ivox]["xSizeVoxel"].asString().c_str()) * CLHEP::cm;
                G4double ySize = atof(jVoxels[ivox]["ySizeVoxel"].asString().c_str()) * CLHEP::cm;
                G4double zSize = atof(jVoxels[ivox]["zSizeVoxel"].asString().c_str()) * CLHEP::cm;
                G4String voxelMat = jVoxels[ivox]["materialVoxel"].asString();
                // Construct the Voxel. At this point only geometry data is stored;
                // no Geant4 objects are created yet (that happens in createG4objects).
                // ivox provides the unique ID; voxelMat is forwarded so each Voxel
                // is self-contained and does not need the material passed again later.
                Voxel *voxel = new Voxel(xPos, yPos, zPos, xSize, ySize, zSize, ivox, voxelMat);
                voxels.push_back(voxel);
            }

            // Summary log: printed in blue.
            G4cout << "\033[1;34m" << "Loaded " << voxels.size()
                   << " voxels from JSON configuration file." << "\033[0m" << G4endl;
        }
    }
    goodGeometry = true;
    G4cout << "DEBUG: Constructor finalizado con éxito." << G4endl;
    Print();
    return;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
bool ConfigurationGeometry::isGood() {
    return goodGeometry;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
G4double ConfigurationGeometry::getSizeX() {
    return uniSizeX;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
G4double ConfigurationGeometry::getSizeY() {
    return uniSizeY;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
G4double ConfigurationGeometry::getSizeZ() {
    return uniSizeZ;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
Detector * ConfigurationGeometry::getDetector(G4int a) {
    return detectors.at(a);
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
G4int ConfigurationGeometry::getNDetectors() {
    return detectors.size();
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
// Returns a pointer to the Voxel at index 'a' in the voxels vector.
// Mirrors getDetector(G4int) in style. Throws std::out_of_range if
// 'a' >= getNVoxels(), which is the safe behaviour from std::vector::at.
//----------------------------------------------------------------------//
Voxel * ConfigurationGeometry::getVoxel(G4int a) {
    return voxels.at(a);
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
// Returns the total number of voxels parsed from TheVoxels in the JSON.
// Returns 0 if no VoxelConfig block was present in the config file.
// Mirrors getNDetectors() in style.
//----------------------------------------------------------------------//
G4int ConfigurationGeometry::getNVoxels() {
    return voxels.size();
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
G4double ConfigurationGeometry::getSizeBoxCRY() {
    return sizeBoxCRY;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
G4double ConfigurationGeometry::getZOffsetCRY() {
    return zOffsetCRY;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//



//----------------------------------------------------------------------//
// createG4Objects                                                      //
//----------------------------------------------------------------------//
void ConfigurationGeometry::createG4objects(G4LogicalVolume *mother, 
                                            std::map<G4String, G4Material *> &materials, 
                                            G4SDManager *SDman) {

    // --- Detectors ---
    // Each detector creates its own G4Box (air) and places its layers inside.
    // The SDManager is passed so layers can attach their G4VSensitiveDetector.
    for(int i = 0; i < detectors.size(); i++) {
        detectors[i]->createG4Objects(G4String(std::to_string(detectors[i]->detId())),
                                      mother, materials, SDman);
    }

    // --- Voxels ---
    // Each voxel creates a G4Box filled with its stored material and places it
    // directly in 'mother' (world logical volume), at the same hierarchy level
    // as the detectors. No SDManager is passed because voxels are passive:
    // they do not register hits, they only act as scattering/absorbing material.
    // If no voxels were defined in the JSON this loop simply does nothing.
    for(int i = 0; i < voxels.size(); i++) {
        voxels[i]->createG4Objects(G4String(std::to_string(voxels[i]->voxelId())),
                                   mother, materials);
    }
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Accesor to class information                                         //
//----------------------------------------------------------------------//
void ConfigurationGeometry::registerCollection(G4String a) {
    collections.push_back(a);
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Print the class information                                          //
//----------------------------------------------------------------------//
void ConfigurationGeometry::Print() {

    G4cout << "\033[1;34m" << "---------------------------------------Geometry information-----------------------------------" << "\033[0m" << G4endl;
    G4cout << "\033[1;34m" << "The loaded geometry is as follows: " << G4endl;
    G4cout << "\033[1;34m" << "The world is contained in [" << -uniSizeX/2.0/CLHEP::cm << ", " << uniSizeX/2.0/CLHEP::cm << "]x[" << -uniSizeY/2.0/CLHEP::cm << ", " << uniSizeY/2.0/CLHEP::cm << "]x[" << -uniSizeZ/2.0/CLHEP::cm << ", " << uniSizeZ/2.0/CLHEP::cm << "]" << G4endl;
    for(G4int i = 0; i < getNDetectors(); i++) {
        detectors.at(i)->Print();
    }
    // Print total voxel count first, then one line per voxel.
    // If no voxels were loaded this prints "Number of voxels: 0" which is
    // useful to confirm the JSON was parsed without a VoxelConfig block.
    G4cout << "\033[1;34m" << "Number of voxels: " << getNVoxels() << "\033[0m" << G4endl;
    for(G4int i = 0; i < getNVoxels(); i++) {
        voxels.at(i)->Print();  // prints id, position (cm) and material
    }
    G4cout << "\033[0m" << G4endl;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//





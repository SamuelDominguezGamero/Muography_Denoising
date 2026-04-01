#include "Layer.hh"
#include "LayerSensor.hh"

//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
Layer::Layer(G4double xPos, G4double yPos, G4double zPos, 
             G4double xRot, G4double yRot, G4double zRot,
             G4double xSize, G4double ySize, G4double zSize,
             G4int ndet, G4int nlayer) :
             GeomObject(xPos, yPos, zPos, xRot, yRot, zRot, xSize, ySize, zSize) {
                ndetId = ndet;
                nlayerId = nlayer;
             };
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// createG4Objects                                              //
//----------------------------------------------------------------------//
void Layer::createG4Objects(G4String name, G4LogicalVolume *mother,
                            std::map<G4String, G4Material*> & materials,
                            G4SDManager *SDman) {

    G4String layerName = G4String("layer_") + name;  
    solidVolume = new G4Box(layerName, sizes[0]/2.0, sizes[1]/2.0, sizes[2]/2.0);
    logicalVolume = new G4LogicalVolume(solidVolume, materials["air"], layerName);
    G4String layerPhysicalName = G4String("layerPhys_") + name;
    physicalVolume = new G4PVPlacement(getRot(), getPos(), 
                                       logicalVolume, layerPhysicalName,
                                       mother, false, 0, true);
    //We need to make this object sensitive
    G4String SDname = layerName;
    G4String Collection = G4String("HitsCollection_") + name;
    LayerSensor *layerSensor = new LayerSensor(SDname = SDname, Collection);
    layerSensor->setLayer(this);
    SDman->AddNewDetector(layerSensor);
    logicalVolume->SetSensitiveDetector(layerSensor);                                   
   
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Return detId                                                         //
//----------------------------------------------------------------------//
G4int Layer::detId() {
	return ndetId;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Return layerId                                                       //
//----------------------------------------------------------------------//
G4int Layer::layerId() {
	return nlayerId;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

//----------------------------------------------------------------------//
// Print                                                                //
//----------------------------------------------------------------------//
void Layer::Print() {

    G4cout << "\033[1;34m" << "Layer" << "\033[0m" << G4endl;
    G4cout << "\033[1;34m" << "Location x: " << pos.x()/CLHEP::cm << ", y: " << pos.y()/CLHEP::cm << ", z: " << pos.z()/CLHEP::cm << G4endl;
    G4cout << "\033[1;34m" << "Rotation x: " << rots.x() << ", y: " << rots.y() << ", z: " << rots.z() << G4endl;
    G4cout << "\033[1;34m" << "Sizes x: " << sizes.x()/CLHEP::cm << ", y: " << sizes.y()/CLHEP::cm << ", z: " << sizes.z()/CLHEP::cm << G4endl;
    G4cout << "\033[0m" << G4endl;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

#ifndef Layer_h
#define Layer_h 1

#include "GeomObject.hh"

class Layer : GeomObject {

public:

    Layer(G4double, G4double, G4double, G4double, G4double, G4double, G4double, G4double, G4double, G4int, G4int);
       
    G4int detId();

    G4int layerId();

    void createG4Objects(G4String, G4LogicalVolume *, std::map<G4String, G4Material*> &, G4SDManager*);
    
    void Print();

private:
    
    G4int ndetId, nlayerId;
};



#endif


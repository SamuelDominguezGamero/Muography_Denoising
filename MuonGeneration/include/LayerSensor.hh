#ifndef LayerSensor_h
#define LayerSensor_h 1

#include "G4VSensitiveDetector.hh"
#include "LayerSensorHit.hh"
#include "Layer.hh"

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;

class LayerSensor : public G4VSensitiveDetector
{

public:
    LayerSensor(G4String name, G4String collection);
    
    void setLayer(Layer *);
    virtual ~LayerSensor();
    Layer *getLayer();
    virtual void Initialize(G4HCofThisEvent*HCE);
    virtual G4bool ProcessHits(G4Step*aStep,G4TouchableHistory*ROhist);
    virtual void EndOfEvent(G4HCofThisEvent*HCE);

private:
    LayerSensorHitsCollection * hitsCollection;
    G4ThreeVector theLocalPosError;
    Layer *layer;
    G4int HCID;
};


#endif


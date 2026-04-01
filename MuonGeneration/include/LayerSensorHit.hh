#ifndef LayerSensorHit_h
#define LayerSensorHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"

class G4AttDef;
class G4AttValue;

class LayerSensorHit : public G4VHit
{
public:

    LayerSensorHit();
    virtual ~LayerSensorHit();
    LayerSensorHit(const LayerSensorHit &right);
    const LayerSensorHit& operator=(const LayerSensorHit &right);
    int operator==(const LayerSensorHit &right) const;

    inline void *operator new(size_t);
    inline void operator delete(void *aHit);

    inline float x();
    inline float y();

    virtual const std::map<G4String,G4AttDef>* GetAttDefs() const;
    virtual std::vector<G4AttValue>* CreateAttValues() const;
    virtual void Print();

private:
    G4int eventNumber;
    G4int detectorID;
    G4int layerID;
    G4double energy;
    G4ThreeVector localPos;
    G4ThreeVector globalPos;
    G4ThreeVector localDir;
    G4ThreeVector globalDir;
    G4int genID;

public:
    inline void SetEventNumber(G4int z) {
        eventNumber = z;
    }
    inline G4int GetEventNumber() const {
        return eventNumber;
    }
    inline void SetDetectorID(G4int z) {
        detectorID = z;
    }
    inline G4int GetDetectorID() const {
        return detectorID;
    }
    inline void SetLayerID(G4int z) {
        layerID = z;
    }
    inline G4int GetLayerID() const {
        return layerID;
    }
    inline void SetEnergy(G4double t) {
        energy = t;
    }
    inline G4double GetEnergy() const {
        return energy;
    }
    inline void SetLocalPos(G4ThreeVector xyz) {
        localPos = xyz;
    }
    inline G4ThreeVector GetLocalPos() const {
        return localPos;
    }
    inline void SetGlobalPos(G4ThreeVector xyz) {
        globalPos = xyz;
    }
    inline G4ThreeVector GetGlobalPos() const {
        return globalPos;
    }
    inline void SetLocalDir(G4ThreeVector xyz) {
        localDir = xyz;
    }
    inline G4ThreeVector GetLocalDir() const {
        return localDir;
    }
    inline void SetGlobalDir(G4ThreeVector xyz) {
        globalDir = xyz;
    }
    inline G4ThreeVector GetGlobalDir() const {
        return globalDir;
    }
    inline void SetGenID(G4int genID_){
        genID = genID_;
    }
    inline G4int GetGenID() const {
        return genID;
    }
};


//----------------------------------------------------------------------//
// Collections of this                                                  //
//----------------------------------------------------------------------//
typedef G4THitsCollection<LayerSensorHit> LayerSensorHitsCollection;

extern G4Allocator<LayerSensorHit> LayerSensorHitAllocator;
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Redefinition of new operator                                         //
//----------------------------------------------------------------------//
inline void* LayerSensorHit::operator new(size_t) {
    void* aHit;
    aHit = (void*)LayerSensorHitAllocator.MallocSingle();
    return aHit;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Redefinition of delete operator                                      //
//----------------------------------------------------------------------//
inline void LayerSensorHit::operator delete(void* aHit) {
    LayerSensorHitAllocator.FreeSingle((LayerSensorHit*) aHit);
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

#endif



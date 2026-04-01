//------------------------------------------------------------//
// Voxel class                                               //
//                                                           //
// Represents a single passive cubic volume in the Geant4   //
// geometry. Voxels do NOT register hits: they act purely    //
// as material blocks where particles (muons) can scatter,   //
// lose energy, etc. according to the active physics list.   //
//                                                           //
// Voxels inherit from GeomObject, exactly like Detector     //
// and Layer, so they share the same position/rotation and   //
// G4 volume pointers infrastructure.                        //
//                                                           //
// Typical use: spell out a word (e.g. "MUON") in lead       //
// between two detector planes to study scattering patterns. //
//------------------------------------------------------------//

#ifndef Voxel_h
#define Voxel_h 1

#include <map>
#include "globals.hh"
#include "GeomObject.hh"       // base class: position, rotation, G4 volume pointers
#include "G4LogicalVolume.hh"  // needed for mother volume argument
#include "G4Material.hh"       // needed for materials map value type

// Voxel inherits privately from GeomObject, same convention as Detector and Layer.
// Private inheritance means GeomObject's public interface is only accessible
// internally, keeping the external API clean.
class Voxel : public GeomObject {

public:

    // Constructor.
    // xPos, yPos, zPos : global position of the voxel centre (in CLHEP units, i.e. mm).
    //                    These are WORLD coordinates, not relative to any parent volume.
    // xSize             : x length of the voxel
    // ySize             : y length of the voxel
    // zSize             : z length of the voxel
    // id               : unique integer index (0, 1, 2, ...) assigned by ConfigurationGeometry
    //                    during JSON parsing. Used for identification and printing.
    // material         : string key that must exist in the materials map passed to
    //                    createG4Objects (e.g. "lead", "iron", "air"). Stored internally
    //                    so each voxel carries its own material and createG4Objects
    //                    does not need it as an external argument.
    Voxel(G4double xPos,
          G4double yPos,
          G4double zPos,
          G4double xSize,
          G4double ySize,
          G4double zSize,
          G4int    id,
          G4String material);

    // Returns the unique index of this voxel (set at construction time).
    G4int voxelId();

    // Builds the Geant4 solid, logical volume and physical volume for this voxel
    // and places it inside 'mother' (typically the world logical volume).
    //
    // Unlike Layer::createG4Objects, this method does NOT attach a G4VSensitiveDetector,
    // so no hits are recorded when particles cross this volume. Physics interactions
    // (scattering, energy loss) still happen normally because Geant4 always applies
    // the registered physics processes regardless of sensitivity.
    //
    // name      : string identifier appended to volume names (e.g. "voxel_3").
    // mother    : logical volume in which this voxel is placed (world logical volume).
    // materials : map from string keys to G4Material pointers, built by DetectorConstruction.
    //             The stored voxelMaterial string is looked up in this map.
    void createG4Objects(G4String name,
                         G4LogicalVolume *mother,
                         std::map<G4String, G4Material*> &materials);

    // Prints voxel id, global position and size to G4cout. Called from
    // ConfigurationGeometry::Print() at startup so the user can verify the geometry.
    void Print();

private:

    // Unique index of this voxel within the TheVoxels array (0-based).
    // Assigned sequentially by ConfigurationGeometry during JSON parsing.
    G4int    nvoxelId;

    // String key of the material for this voxel (e.g. "lead").
    // Must match a key in the materials map provided by DetectorConstruction.
    // Stored here so the voxel is self-contained and createG4Objects needs
    // no external material argument.
    G4String voxelMaterial;
    G4double fXSize;
    G4double fYSize;
    G4double fZSize;
};

#endif

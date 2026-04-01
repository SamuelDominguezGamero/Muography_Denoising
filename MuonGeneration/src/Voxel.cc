#include "Voxel.hh"

#include "G4Box.hh"            // G4Box solid: the cubic shape of each voxel
#include "G4LogicalVolume.hh"  // G4LogicalVolume: associates solid + material
#include "G4PVPlacement.hh"    // G4PVPlacement: places the logical volume in the world
#include "G4SystemOfUnits.hh"  // CLHEP unit system (cm, mm, deg, ...)


//----------------------------------------------------------------------//
// Constructor                                                          //
//----------------------------------------------------------------------//
// Delegates position/rotation/size storage to GeomObject.
// Rotation is always (0,0,0): voxels are axis-aligned cubes.
// The same 'size' value is passed for all three dimensions to enforce
// the cubic shape enforced by the VoxelConfig JSON block.
// 'material' is stored as a string key and resolved to a G4Material*
// later in createG4Objects, when the materials map is available.
//----------------------------------------------------------------------//
Voxel::Voxel(G4double xPos, 
             G4double yPos,
             G4double zPos,
             G4double xSize,
             G4double ySize,
             G4double zSize,
             G4int    id,
             G4String material)
    : GeomObject(xPos, yPos, zPos,
                 0, 0, 0,          // no rotation: voxels are always axis-aligned
                 xSize, ySize, zSize),
      nvoxelId(id),
      voxelMaterial(material),
      fXSize(xSize),
      fYSize(ySize),
      fZSize(zSize)
{}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Accessor: voxelId                                                    //
//----------------------------------------------------------------------//
// Returns the unique 0-based index of this voxel.
// Used in createG4Objects to build unique volume names ("voxel_0",
// "voxel_1", ...) and in ConfigurationGeometry::Print() for logging.
//----------------------------------------------------------------------//
G4int Voxel::voxelId() {
    return nvoxelId;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// createG4Objects                                                      //
//----------------------------------------------------------------------//
// Builds the three Geant4 objects that represent this voxel in the
// simulation geometry:
//
//   1. G4Box (solidVolume)       — defines the cubic shape.
//   2. G4LogicalVolume           — binds the shape to its material.
//                                  No sensitive detector is attached here,
//                                  which is the key difference from Layer.
//   3. G4PVPlacement             — places the volume at the world position
//                                  stored in GeomObject::pos.
//
// Because no G4VSensitiveDetector is set on the logical volume, Geant4
// will NOT call ProcessHits when a particle crosses this voxel. However,
// all physics processes (multiple Coulomb scattering, ionisation, etc.)
// still apply normally, since they are governed by the physics list, not
// by sensitivity.
//----------------------------------------------------------------------//
void Voxel::createG4Objects(G4String name,
                             G4LogicalVolume *mother,
                             std::map<G4String, G4Material*> &materials) {

    // Build a unique name for all G4 objects of this voxel.
    // 'name' is the string representation of nvoxelId, e.g. "3".
    G4String voxName = "voxel_" + name;

    // --- Solid ---
    // G4Box takes half-lengths, so we divide each size by 2.
    solidVolume = new G4Box(voxName,
                            fXSize / 2.0,
                            fYSize / 2.0,
                            fZSize / 2.0);

    // --- Logical volume ---
    // Look up the G4Material* from the materials map using the string key
    // stored at construction time (e.g. "lead" → G4_Pb).
    // No sensitive detector is set → purely passive volume.
    logicalVolume = new G4LogicalVolume(solidVolume,
                                        materials[voxelMaterial],
                                        voxName);

    // --- Physical placement ---
    // getRot() returns the G4RotationMatrix* from GeomObject (identity for voxels).
    // getPos() returns the G4ThreeVector world position from GeomObject.
    // The voxel is placed directly inside 'mother' (the world logical volume),
    // at the same hierarchy level as the detectors.
    physicalVolume = new G4PVPlacement(getRot(),       // rotation (identity)
                                       getPos(),       // global position
                                       logicalVolume,  // what to place
                                       "voxelPhys_" + name, // physical volume name
                                       mother,         // parent volume (world)
                                       false,          // no boolean operation
                                       0,              // copy number
                                       false);          // check for overlaps ---> can cause bugs
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


//----------------------------------------------------------------------//
// Print                                                                //
//----------------------------------------------------------------------//
// Prints a one-line summary of this voxel to G4cout.
// Called from ConfigurationGeometry::Print() at startup.
// Converts from CLHEP internal units (mm) back to cm for readability.
//----------------------------------------------------------------------//
void Voxel::Print() {
    G4cout << "  Voxel " << nvoxelId
           << "  pos=("   << pos[0]    / CLHEP::cm << ", "
                          << pos[1]    / CLHEP::cm << ", "
                          << pos[2]    / CLHEP::cm << ") cm"
           << "  xSize="   << fXSize  / CLHEP::cm << " cm"
           << "  ySize="   << fYSize  / CLHEP::cm << " cm"
           << "  zSize="   << fZSize  / CLHEP::cm << " cm"
           << "  mat="    << voxelMaterial
           << G4endl;
}
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//

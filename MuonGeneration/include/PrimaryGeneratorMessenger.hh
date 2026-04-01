//----------------------------------------------------------------------//
// ___  ___                    _____           _                        //
// |  \/  |                   /  ___|         | |                       //
// | .  . |_   _  ___  _ __   \ `--. _   _ ___| |_ ___ _ __ ___  ___    //
// | |\/| | | | |/ _ \| '_ \   `--. \ | | / __| __/ _ \ '_ ` _ \/ __|   //
// | |  | | |_| | (_) | | | | /\__/ / |_| \__ \ ||  __/ | | | | \__ \   //
// \_|  |_/\__,_|\___/|_| |_| \____/ \__, |___/\__\___|_| |_| |_|___/   //
//                                    __/ |                             //
//----------------------------------------------------------------------//
// A project by: C. Diez, P. Gomez and P. Martinez                      //
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//
// PrimaryGeneratorMessanger.hh                                         //
//----------------------------------------------------------------------//
// Class to organize the output produced by the generator.              //
//----------------------------------------------------------------------//
//----------------------------------------------------------------------//


#ifndef PrimaryGeneratorMessenger_h
#define PrimaryGeneratorMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

class PrimaryGeneratorAction;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithoutParameter;


class PrimaryGeneratorMessenger: public G4UImessenger {
public:
    PrimaryGeneratorMessenger(PrimaryGeneratorAction *);
    ~PrimaryGeneratorMessenger();

    void SetNewValue(G4UIcommand*, G4String);

private:
    PrimaryGeneratorAction*      Action;
    G4UIdirectory*               CRYDir;
    G4UIcmdWithAString*          FileCmd;
    G4UIcmdWithAString*          InputCmd;
    G4UIcmdWithoutParameter*     UpdateCmd;
    std::string* MessInput;
};


#endif


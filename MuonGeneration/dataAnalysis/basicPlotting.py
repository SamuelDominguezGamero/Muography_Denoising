import json, sys, optparse
import ROOT as r
from array import array
from tools.tdrstyle import tdrstyle
import math
import sys

def save(name, histoOpen, histoPyr, option):
  
    filename = name + '.png'
    canvasname = 'can_' + name
    c = r.TCanvas(canvasname)
    c.SetLogz(r.kTRUE)
    c.Divide(3,1)
    c.cd(1)
    histoOpen.Draw(option)
    c.cd(2)
    histoPyr.Draw(option)
    c.cd(3)
    histoNew = histoPyr.Clone('histoNew')
    #histoNew.Add(histoOpen, -1.0)
    histoNew.Divide(histoOpen)
    #histoNew.GetZaxis().SetRangeUser(0.0, 1.0)
    histoNew.Draw(option)
    c.SaveAs(filename)



def loop(thefile, tag):
  
    #'atan2(-vy, -vx):acos(-vz/(sqrt(vx*vx+vy*vy+vz*vz))'
    histos = dict()
    histos['phitheta'] = (r.TH2F('phitheta_' + tag, '', 90, -90, 90, 90, 0, 90), '180.0/3.14159 * acos(-vz/sqrt(vx*vx+vy*vy+vz*vz)): 180.0/3.141592 * atan2(-vy, -vx)', '', 'COLZ')
    #histos['phitheta'][0].GetZaxis().SetRangeUser(0, 800)
    histos['phitheta'][0].GetXaxis().SetTitle("#phi [rad]")
    histos['phitheta'][0].GetYaxis().SetTitle("#theta [rad]")
    histos['yzproj'] = (r.TH2F('yzproj_' + tag, '', 100, -300, 300, 100, 0, 200), '(z+(-1100.0-x)*(vz/vx))/100.0:(y+(-1100.0-x)*(vy/vx))/100.0', '', 'COLZ')
    #histos['yzproj'][0].GetZaxis().SetRangeUser(0, 800)
    histos['yzproj'][0].GetXaxis().SetTitle("y [m]")
    histos['yzproj'][0].GetYaxis().SetTitle("z [m]")
    histos['yzprojres'] = (r.TH2F('yzprojres_' + tag, '', 20, -35, 35, 30, 30, 100), '(z+(-1100.0-x)*(vz/vx))/100.0:(y+(-1100.0-x)*(vy/vx))/100.0', '', 'COLZ')
    #histos['yzproj'][0].GetZaxis().SetRangeUser(0, 800)
    histos['yzprojres'][0].GetXaxis().SetTitle("y [m]")
    histos['yzprojres'][0].GetYaxis().SetTitle("z [m]")

    t = thefile.events

    #for ev in thefile.events:
    #    histos['phitheta'][0].Fill(math.atan2(-ev.vy, -ev.vx), math.acos(-ev.vz/math.sqrt(ev.vx*ev.vx+ev.vy*ev.vy+ev.vz*ev.vz)))
    for n, l in histos.items():
        t.Project(l[0].GetName(), l[1], l[2])
        print(l[0].GetName())
        print(l[0].GetEntries())    
        l[0].Scale(1.0/l[0].GetEntries())
        l[0].SetStats(False)
      
    return histos



if __name__ == '__main__':

    parser = optparse.OptionParser(usage='usage: %prog [options] path', version='%prog 1.0')
    parser.add_option('-o', '--opensky', action='store', type='string', dest='openskyFile', default='opensky.root', help='Open sky ROOT File')
    parser.add_option('-p', '--pyramid', action='store', type='string', dest='pyramidFile', default='pyramid.root', help='Pyramid ROOT File')
    (opts, args) = parser.parse_args()

    tdr = tdrstyle()
    tdr.tdrStyle.SetPalette(r.kSunset)    
    #Location of the chamber
    xchamber = [-11.1316, -2.2778, 54.3354]
    xdetector = [-120.0, 0, 1.55]
    thetaMax = math.atan2(64.3354-1.55, 120.0 - 11.1316)
    thetaMin = math.atan2(44.3354-1.55, 120.0 - 11.1316)
    phiMax = math.atan2(-2.2778+10.0, 120 - 11.1316)
    phiMin = math.atan2(-2.2778-10.0, 120 - 11.1316)

    #thetaMax = math.pi/2.0
    #thetaMin = 0
    #phiMax = math.pi/2.0
    #phiMin = -math.pi/2.0

    print('thetaMax is:', thetaMax, 'thetaMin is:', thetaMin)
    print('phiMax is:', phiMax, 'phiMin is:', phiMin)


    try:
        inputO = r.TFile(opts.openskyFile)
    except:
        print('Cannot open opensky file')
        sys.exit()

    try:
        inputP = r.TFile(opts.pyramidFile)
    except:
        print('Cannot open pyramid file')
        sys.exit()


    histosOpensky = loop(inputO, 'opensky')
    histosPyramid = loop(inputP, 'pyramid')


    for n, l in histosOpensky.items():

        save(n, l[0], histosPyramid[n][0], l[3])


    inputO.Close()
    inputP.Close()




   



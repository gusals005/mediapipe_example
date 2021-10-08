import sys
import imageconverter as c
import jsonmaker as j
import camconverter as cam

def main(argv):
  mode = argv[1]
  if len(argv) > 2:
    filename = argv[2]
  
  if int(mode) == 0 :
    c.imageConvert(filename) 
  elif int(mode) == 1 :
    j.convertVideotoJson(filename)
  else :
    cam.convertKeypointsfromCam(filename)
  

if __name__ == "__main__":
  main(sys.argv)


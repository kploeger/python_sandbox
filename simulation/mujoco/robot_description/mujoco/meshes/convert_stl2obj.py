import Mesh
import os

for entry in os.scandir(os.cwd()):
    importPath = entry.path
    if ".stl" in importPath:
        exportPath = importPath.rsplit('.', 1)[0] + '.obj'
        exportPath = exportPath.replace(inputFolder, outputFolder)
        mesh = Mesh.Mesh() # creates an instance of the Mesh object
        mesh.read(importPath) # read STL
        mesh.write(exportPath) # write OBJ
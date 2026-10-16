"""Render the actual adapter and camera on supplied robot geometry.
First run export_robot_preview.py with the project's MuJoCo environment.
Then run this script with the CAD environment. No control interfaces are imported.
"""
from pathlib import Path
import os
os.environ.setdefault('DISPLAY',':1')
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
import cadquery as cq
ROOT=Path(__file__).resolve().parent
D=np.load('/tmp/zed2i_robot_geometry.npz')
R=D['mount_rotation'];O=D['mount_origin']

def mesh_actor(v,f,color):
 points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.ascontiguousarray(v),deep=True))
 cells=vtk.vtkCellArray();cells.SetCells(len(f),numpy_to_vtkIdTypeArray(np.column_stack((np.full(len(f),3),f)).astype(np.int64).ravel(),deep=True))
 p=vtk.vtkPolyData();p.SetPoints(points);p.SetPolys(cells)
 normals=vtk.vtkPolyDataNormals();normals.SetInputData(p);normals.SetFeatureAngle(40)
 mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(normals.GetOutputPort())
 a=vtk.vtkActor();a.SetMapper(mapper);a.GetProperty().SetColor(*color)
 a.GetProperty().SetAmbient(.25);a.GetProperty().SetDiffuse(.75);a.GetProperty().SetSpecular(.12)
 return a

actors=[]
for i in range(int(D['count'])):
 color=D[f'c{i}'].copy()
 # Make untextured stand geometry legible without changing its shape.
 actors.append(mesh_actor(D[f'v{i}'],D[f'f{i}'],color))

def cad_actor(shape,color):
 verts,faces=shape.tessellate(.12,.15)
 v=np.array([x.toTuple() for x in verts])/1000
 return mesh_actor(v@R.T+O,np.asarray(faces),color)

adapter=cq.importers.importStep(str(ROOT/'output/zed2i_adapter_assembly_orientation.step')).val()
camera=(cq.importers.importStep(str(ROOT/'references/ZED2i.step'))
        .translate((36.6538415,17.299,8.73585029186822))
        .rotate((0,0,0),(0,0,1),180).translate((0,25,16))).val()
actors.extend([cad_actor(adapter,(.03,.7,.86)),cad_actor(camera,(.19,.22,.25))])
# Screw heads and washers are schematic, only showing installation locations.
for x in (-22.5,22.5):
 washer=cq.Solid.makeCylinder(3.5,.5,cq.Vector(x,6,0),cq.Vector(0,1,0))
 head=cq.Solid.makeCylinder(2.75,3,cq.Vector(x,6.5,0),cq.Vector(0,1,0))
 actors.append(cad_actor(washer.fuse(head),(.6,.62,.64)))
for x in (-18,18):
 washer=cq.Solid.makeCylinder(3.5,.5,cq.Vector(x,25,10),cq.Vector(0,0,-1))
 head=cq.Solid.makeCylinder(2.75,3,cq.Vector(x,25,9.5),cq.Vector(0,0,-1))
 actors.append(cad_actor(washer.fuse(head),(.6,.62,.64)))
window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(2000,1400);window.SetMultiSamples(8)
views=[((0,0,.52,1),(1.5,1.5,1.0),(.08,0,.35),.48,'ROBOT OVERVIEW'),
       ((.52,.5,1,1),(.48,.38,.89),(.10,0,.66),.135,'MOUNT CLOSE-UP'),
       ((.52,0,1,.5),(.21,.65,.70),(.105,0,.66),.105,'SIDE VIEW')]
for viewport,eye,look,scale,title in views:
 renderer=vtk.vtkRenderer();renderer.SetViewport(*viewport);renderer.SetBackground(.95,.965,.98);window.AddRenderer(renderer)
 for actor in actors:renderer.AddActor(actor)
 cam=renderer.GetActiveCamera();cam.SetPosition(*eye);cam.SetFocalPoint(*look);cam.SetViewUp(0,0,1);cam.ParallelProjectionOn();cam.SetParallelScale(scale)
 if viewport[0]==0:
  renderer.ResetCamera();cam.SetParallelScale(cam.GetParallelScale()*1.08)
 label=vtk.vtkTextActor();label.SetInput(title);label.SetPosition(25,1320 if viewport[3]-viewport[1]>0.5 else 630)
 label.GetTextProperty().SetFontSize(26);label.GetTextProperty().SetColor(.1,.17,.22);renderer.AddActor2D(label)
 note=vtk.vtkTextActor();note.SetInput('CYAN: printed adapter | BLACK: ZED 2i' if viewport[0]==0 else 'Illustrative tilt setting; not a physical fit test')
 note.SetPosition(22,22);note.GetTextProperty().SetFontSize(19);note.GetTextProperty().SetColor(.12,.2,.25);note.GetTextProperty().SetBackgroundColor(.95,.965,.98);note.GetTextProperty().SetBackgroundOpacity(.95);renderer.AddActor2D(note)
window.Render();capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.Update()
writer=vtk.vtkPNGWriter();writer.SetFileName(str(ROOT/'output/on_robot.png'));writer.SetInputConnection(capture.GetOutputPort());writer.Write();window.Finalize()
print(ROOT/'output/on_robot.png')

"""ZED 2i bracket CAD construction. All CAD dimensions are millimetres.
Run with Python 3.10+, cadquery 2.7, trimesh and matplotlib.
"""
from pathlib import Path
import json
import cadquery as cq
import trimesh

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
P = dict(width=64.0, stand_height=20.0, stand_pitch=45.0,
         wall=6.0, shelf_depth=46.0, shelf_top=16.0,
         camera_pitch=36.0, camera_y=25.0, hole_diameter=3.4,
         slot_length=5.4, gusset_width=4.0)

def build():
    # Assembly coordinates: X across camera, Y forward, Z up.
    # Existing metal mounting face is Y=0, with hole centres Z=0.
    back = cq.Workplane('XY').box(P['width'], P['wall'], P['stand_height']).translate((0,3,0))
    shelf = cq.Workplane('XY').box(P['width'], P['shelf_depth'], P['wall']).edges('|Z').fillet(2).translate((0,23,13))
    part = back.union(shelf)
    for x in (-32,28):
        rib = cq.Workplane('YZ', origin=(x,0,0)).polyline([(5,-9),(39,10),(5,11)]).close().extrude(P['gusset_width'])
        part = part.union(rib)
    # Short horizontal slots absorb measurement / printing error.
    for x in (-P['stand_pitch']/2, P['stand_pitch']/2):
        cutter = cq.Workplane('XZ', origin=(x,7,0)).slot2D(P['slot_length'],P['hole_diameter']).extrude(8)
        part = part.cut(cutter)
    for x in (-P['camera_pitch']/2,P['camera_pitch']/2):
        part = part.cut(cq.Workplane('XY',origin=(x,P['camera_y'],9)).circle(P['hole_diameter']/2).extrude(8))
    return part.clean()

def export(part,name):
    cq.exporters.export(part,str(OUT/f'{name}.step'))
    cq.exporters.export(part,str(OUT/f'{name}.stl'),tolerance=0.03,angularTolerance=0.08)

part=build()
export(part,'zed2i_adapter_assembly_orientation')
# Print on the left side: no support under the shelf; leave STL orientation as supplied.
print_part=part.rotate((0,0,0),(0,1,0),-90).translate((16,0,32))
export(print_part,'zed2i_adapter_PRINT')
coupon=cq.Workplane('XY').box(64,20,2,centered=(True,True,False))
for x in (-22.5,22.5):
    coupon=coupon.cut(cq.Workplane('XY',origin=(x,0,-1)).slot2D(5.4,3.4).extrude(4))
export(coupon,'stand_fit_coupon_PRINT')

# Official STEP is aligned by its bottom mounting axes; front faces +Y.
reference=cq.importers.importStep(str(ROOT/'references/ZED2i.step'))
camera=(reference.translate((36.6538415,17.299,-(-8.73585029186822)))
        .rotate((0,0,0),(0,0,1),180).translate((0,P['camera_y'],P['shelf_top'])))
# Simplified plate only. Its thickness is an assumption, not measured geometry.
metal=cq.Workplane('XY').box(68,2,20).translate((0,-1,0))
for x in (-22.5,22.5):
    metal=metal.cut(cq.Workplane('XZ',origin=(x,1,0)).circle(1.5).extrude(4))
assy=cq.Assembly(name='ZED2i_mount_preview')
assy.add(part,name='PRINTED_ADAPTER',color=cq.Color(0.1,0.65,0.8))
assy.add(camera,name='OFFICIAL_ZED2i',color=cq.Color(0.2,0.2,0.2))
assy.add(metal,name='METAL_PLATE_APPROXIMATION',color=cq.Color(0.4,0.4,0.4))
assy.export(str(OUT/'assembly_preview.step'))
# Check screw / driver envelopes against the plastic, excluding the intended bore.
checks={}
for x in (-22.5,22.5):
    tool=cq.Solid.makeCylinder(3.5,60,cq.Vector(x,6.01,0),cq.Vector(0,1,0))
    checks[f'stand_driver_{x}_interference_mm3']=part.val().intersect(tool).Volume()
for x in (-18,18):
    tool=cq.Solid.makeCylinder(3.5,50,cq.Vector(x,25,9.99),cq.Vector(0,0,-1))
    checks[f'camera_driver_{x}_interference_mm3']=part.val().intersect(tool).Volume()
checks['camera_body_interference_mm3']=part.val().intersect(camera.val()).Volume()
checks['stand_plate_interference_mm3']=part.val().intersect(metal.val()).Volume()
checks['cad_valid']=part.val().isValid()
checks['solid_count']=len(part.solids().vals())
mesh=trimesh.load(OUT/'zed2i_adapter_PRINT.stl')
checks.update(stl_watertight=bool(mesh.is_watertight),stl_winding_consistent=bool(mesh.is_winding_consistent),
              stl_extents_mm=mesh.extents.tolist(),volume_cm3=part.val().Volume()/1000,
              assumptions=['Stand holes centred vertically on 20 mm plate.','Stand metal plate preview thickness 2 mm; not measured.'],
              limitations=['No physical fit, load, thermal or vibration test.','Full tilt sweep and cable clearance require physical check.'])
(OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
(OUT/'parameters.json').write_text(json.dumps(P,indent=2)+'\n')
for k,v in checks.items(): print(k,v)
assert checks['cad_valid'] and checks['solid_count']==1 and checks['stl_watertight']
assert all(v<0.01 for k,v in checks.items() if 'interference_mm3' in k)
# Render actual triangulated geometry with a depth buffer (no generated imagery).
import os
os.environ.setdefault('DISPLAY', ':1')
import vtk

def actor_for(shape, color):
    verts, faces = shape.tessellate(0.10,0.15)
    points = vtk.vtkPoints()
    for v in verts: points.InsertNextPoint(*v.toTuple())
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3)
        for i in face: cells.InsertCellPoint(i)
    poly = vtk.vtkPolyData(); poly.SetPoints(points); poly.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals(); normals.SetInputData(poly); normals.SetFeatureAngle(40)
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color); actor.GetProperty().SetAmbient(0.22)
    actor.GetProperty().SetDiffuse(0.78); actor.GetProperty().SetSpecular(0.15)
    return actor

window = vtk.vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1800,900)
for i in range(2):
    renderer = vtk.vtkRenderer(); renderer.SetViewport(i/2,0,(i+1)/2,1)
    renderer.SetBackground(0.96,0.97,0.98); window.AddRenderer(renderer)
    renderer.AddActor(actor_for(part.val(),(0.05,0.62,0.77)))
    renderer.AddActor(actor_for(metal.val(),(0.42,0.44,0.47)))
    if i: renderer.AddActor(actor_for(camera.val(),(0.24,0.27,0.30)))
    cam=renderer.GetActiveCamera(); cam.SetPosition(110,160,-35 if not i else 95); cam.SetFocalPoint(0,20,4 if not i else 20)
    cam.SetViewUp(0,0,1); cam.ParallelProjectionOn(); cam.SetParallelScale(47 if not i else 85)
    title=vtk.vtkTextActor(); title.SetInput('Adapter underside | 64 x 46 x 26 mm' if not i else 'Assembly | official ZED 2i model')
    title.SetPosition(25,835); title.GetTextProperty().SetFontSize(27); title.GetTextProperty().SetColor(0.1,0.15,0.2)
    renderer.AddActor2D(title)
    note=vtk.vtkTextActor(); note.SetInput('45 mm M3 stand pattern / 36 mm M3 camera pattern' if not i else 'Proxy metal plate only; full tilt clearance needs fit test')
    note.SetPosition(25,40); note.GetTextProperty().SetFontSize(20); note.GetTextProperty().SetColor(0.2,0.25,0.3)
    renderer.AddActor2D(note)
window.Render()
capture=vtk.vtkWindowToImageFilter(); capture.SetInput(window); capture.Update()
writer=vtk.vtkPNGWriter(); writer.SetFileName(str(OUT/'preview.png')); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
window.Finalize()

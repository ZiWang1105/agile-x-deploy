"""Chinese side-section installation schematic, deliberately not to scale."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import math
OUT=Path(__file__).resolve().parent/'output'
im=Image.new('RGB',(1600,1120),'#f6f8fb');d=ImageDraw.Draw(im)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
def text(x,y,s,size=28,color='#172c3c',bold=False):
 d.text((x,y),s,font=ImageFont.truetype(BOLD if bold else FONT,size),fill=color,spacing=14)
def arrow(a,b,color,width=7):
 d.line([a,b],fill=color,width=width)
 dx,dy=b[0]-a[0],b[1]-a[1];n=math.hypot(dx,dy);ux,uy=dx/n,dy/n
 d.polygon([b,(b[0]-22*ux+10*uy,b[1]-22*uy-10*ux),(b[0]-22*ux-10*uy,b[1]-22*uy+10*ux)],fill=color)
blue='#0a9fbe';orange='#db701b';green='#23894c'
text(55,35,'这个打印件，就是一个 L 形托架',46,bold=True)
text(55,105,'现有黑色板保留。先把打印件拧在黑色板上，再把相机拧在托板上。',29)
d.rounded_rectangle((35,180,930,930),radius=20,fill='white')
text(65,205,'从侧面看',29,bold=True)
text(65,251,'每组只画 1 颗螺丝，实际每组各 2 颗。',23,color='#60717b')
# Side section, with ribs omitted to expose screw paths.
d.rectangle((244,455,270,685),fill='#424b54')
d.polygon([(270,410),(770,410),(770,455),(315,455),(315,685),(270,685)],fill=blue)
# Camera sits on the horizontal shelf.
d.rounded_rectangle((245,305,810,410),radius=18,fill='#28323c')
text(405,326,'ZED 2i 相机',35,color='white',bold=True)
# Stand screw: head and washer on printed-part side, shaft toward metal.
d.rectangle((225,568,360,580),fill=orange)
d.rectangle((316,558,324,591),fill=orange)
d.rounded_rectangle((325,553,356,595),radius=4,fill=orange)
arrow((450,575),(370,575),orange)
# Camera screw: inserted from below; section shows engagement in camera.
d.rectangle((567,380,579,466),fill=green)
d.rectangle((554,456,591,465),fill=green)
d.rounded_rectangle((550,466,596,496),radius=4,fill=green)
arrow((573,595),(573,525),green)
text(62,471,'现有的\n黑色金属板',26)
d.line([(195,522),(255,522)],fill='#61717c',width=3)
text(402,625,'① 朝黑色板拧',27,color=orange,bold=True)
text(620,495,'② 向上拧\n进相机底部',27,color=green,bold=True)
text(365,730,'蓝色部分 = 要打印的件',30,color=blue,bold=True)
text(65,830,'为了看清螺丝，图中省略两侧加强筋。\n这是连接关系示意图，不按实际比例绘制。',23,color='#60717b')
# Ordered instructions.
text(985,205,'① 先固定打印件',32,color=orange,bold=True)
text(985,260,'支架侧：2 颗 M3 + 垫圈\n两个孔中心距 45 mm。',27)
text(985,355,'螺丝从蓝色件这一侧，\n朝黑色板方向穿过去。',27)
text(985,445,'黑色板若是光孔，背面加螺母；\n若有 M3 螺纹，就拧入原螺纹。',23,color='#526470')
text(985,565,'② 再固定相机',32,color=green,bold=True)
text(985,620,'相机侧：2 颗 M3 × 12 + 垫圈\n两个孔中心距 36 mm。',27)
text(985,715,'从托板下面向上拧，\n拧进 ZED 2i 底部螺纹孔。',27)
text(985,815,'不要把支架侧和相机侧的孔\n当成同一组贯通孔。',24,color='#526470')
d.rounded_rectangle((35,970,1565,1080),radius=16,fill='#e6eef4')
text(65,989,'③ 装好后，用原金属支架两侧的调节螺丝，调整相机俯仰角度。',29,bold=True)
text(65,1035,'相机螺丝实际旋入深度须小于 10 mm；请先用试装片检查支架孔位。',23,color='#526470')
im.save(OUT/'how_to_install.png')
print(OUT/'how_to_install.png')

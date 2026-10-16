"""Dimensioned reference views; dimensions in mm, not a certified shop drawing."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
OUT=Path(__file__).resolve().parent/'output'
fig,axs=plt.subplots(1,3,figsize=(16,6))
blue='#119bbb'
def rect(ax,x,y,w,h,**kw): ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.7,**kw))
def dim(ax,a,b,label,off=0):
    ax.annotate('',xy=a,xytext=b,arrowprops=dict(arrowstyle='<->',color='#425466',lw=1))
    ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+off,label,ha='center',va='bottom',fontsize=11,color='#263545',bbox=dict(facecolor='white',edgecolor='none',pad=1))
a=axs[0];a.set_title('FRONT | stand mounting holes',fontsize=13)
rect(a,-32,-10,64,20,color=blue);rect(a,-32,10,64,6,color=blue)
for x in (-22.5,22.5):
 a.add_patch(FancyBboxPatch((x-2.7,-1.7),5.4,3.4,boxstyle='round,pad=0,rounding_size=1.7',fill=False,color=blue,lw=1.5))
 a.plot([x,x],[-2,-18],color='gray',lw=.7)
dim(a,(-22.5,-16),(22.5,-16),'45',1)
dim(a,(-32,25),(32,25),'64',1)
dim(a,(39,-10),(39,10),'20',1)
a.text(0,-29,'2 x slots 5.4 x 3.4\nHole row at mid-height (assumed)',ha='center',fontsize=10)
a.set(xlim=(-44,47),ylim=(-36,34))
a=axs[1];a.set_title('TOP | camera bottom mounting',fontsize=13)
rect(a,-32,0,64,46,color=blue)
a.plot([-32,32],[6,6],ls='--',color='gray',lw=1)
for x in (-18,18):
 a.add_patch(Circle((x,25),1.7,fill=False,color=blue,lw=1.5))
 a.plot([x,x],[27,37],color='gray',lw=.7)
dim(a,(-18,35),(18,35),'36',1)
dim(a,(39,0),(39,25),'25',1)
dim(a,(-39,0),(-39,46),'46',1)
a.text(0,-10,'2 x diameter 3.4 through\nCamera front faces upward on this view',ha='center',fontsize=10)
a.set(xlim=(-48,48),ylim=(-19,56))
a=axs[2];a.set_title('SIDE | outward is right',fontsize=13)
rect(a,0,-10,6,20,color=blue);rect(a,0,10,46,6,color=blue)
a.add_patch(Polygon([(5,-9),(39,10),(5,11)],fill=False,edgecolor=blue,lw=1.5))
rect(a,-2,-10,2,20,color='#777777')
a.plot([-5,9],[0,0],color='gray',ls='-.',lw=.8)
a.plot([25,25],[9,17],color='gray',ls='--',lw=1)
dim(a,(52,10),(52,16),'6',1)
dim(a,(0,24),(46,24),'46',1)
dim(a,(-10,-10),(-10,16),'26',1)
a.text(23,-24,'Backplate / shelf: 6 thick\nSide ribs: 4 thick\nGrey plate thickness is illustrative',ha='center',fontsize=10)
a.set(xlim=(-20,65),ylim=(-36,34))
for a in axs:
 a.set_aspect('equal');a.axis('off')
fig.suptitle('ZED 2i / AgileX adapter v1 | all dimensions in mm',fontsize=19,y=.97)
fig.text(.5,.04,'Fit prototype. Confirm stand hole height, fastener length, cable clearance and tilt range on the physical bracket.',ha='center',fontsize=11)
fig.subplots_adjust(wspace=.18,top=.84,bottom=.16)
fig.savefig(OUT/'dimensions.png',dpi=160,facecolor='white');plt.close(fig)

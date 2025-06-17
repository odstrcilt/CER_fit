import pylab as plt
from IPython import embed
class DraggableColorbar(object):
    def __init__(self, cbar, mappable):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if hasattr(getattr(plt.cm,i),'N')])
        self.index = self.cycle.index(cbar.cmap.name)

    def connect(self):
        """connect to all the events we need"""
        #embed()
        self.cidpress = self.cbar.ax.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        #self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            #'key_press_event', self.key_press)

    def on_press(self, event):
        """on button press we will see if the mouse is over us and store some data"""
        if event.inaxes != self.cbar.ax: return
        self.press = event.ydata

    def key_press(self, event):
        if event.key=='down':
            self.index += 1
        elif event.key=='up':
            self.index -= 1
        if self.index<0:
            self.index = len(self.cycle)
        elif self.index>=len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.cmap=cmap
        #self.cbar.draw_all()
        self.mappable.set_cmap(cmap)
        self.cbar.get_axes().set_title(cmap)
        self.cbar.ax.figure.canvas.draw()

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return

        yprev = self.press
        ylim = self.cbar.ax.get_ylim()
        dy = (event.ydata - yprev)/(ylim[1]-ylim[0])
        self.press = event.ydata

        

        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        cmin,cmax = self.cbar.ax.get_ylim()
        
        if event.button==1:
            self.cbar.norm.vmax -=   scale*dy 

        if event.button==3:
            self.cbar.norm.vmin -=   scale*dy 


        #self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.ax.figure.canvas.draw()


    def on_release(self, event):
        """on release we reset the press data"""
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.ax.figure.canvas.draw()

    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.cbar.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.ax.figure.canvas.mpl_disconnect(self.cidmotion)

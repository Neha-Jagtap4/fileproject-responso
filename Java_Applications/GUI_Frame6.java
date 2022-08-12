import java.awt.*;
import java.awt.event.*;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.MediaTracker;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowListener;
import java.io.IOException;

class GUI_Frame6 extends Frame{
	static Image im; //create image class object

	public GUI_Frame6(){
		addWindowListener(new WindowAdapter(){
			public void WindowClosing(WindowEvent obj){
				System.exit(0);
			}
		});
	

	im = Toolkit.getDefaultToolkit().getImage("Marvellous Infosystem logofinal.jpg");
	MediaTracker m = new MediaTracker(this);
	m.addImage(im,1);
	try{
		m.waitForID(1);
	}
	catch(InterruptedException obj){}
	}

	public void paint(Graphics g){
		g.drawImage(im,0,0,null);
	}

	public static void main(String arg[]){
		GUI_Frame6 f = new GUI_Frame6();
		f.setTitle("Marvellous infosystem");
		f.setSize(2424,960);
		f.setVisible(true);
	}
}

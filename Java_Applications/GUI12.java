import java.awt.*;
//import java.awt.event.*;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowListener;

class GUI12 extends Frame{
	
	
	GUI12(){
		addWindowListener(new WindowAdapter());}
	
	public void WindowClosing(WindowEvent obj){
		System.exit(0);
		}
	}

	
	a = new Button("Demo");
	b = new Button("Hello");

	a.setBounds(10,100,50,50);
	b.setBounds(100,100,50,50);

	this.add(a);
	this.add(b);

	public static void main(String arg[]){
		GUI12 f = new GUI12();
		f.setTitle("mARVELLOUS iNFOSYSTEM");
		f.setSize(400,400);
		f.setVisible(true);

	}

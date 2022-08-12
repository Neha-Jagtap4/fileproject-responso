import java.awt.*;
//import java.event.*;
import java.awt.event.WindowEvent;
//import java.awt.event.ActionListener;
import java.awt.event.WindowListener;
import java.io.*;

class GUI_Frame1 extends Frame{
	GUI_Frame1(String str){
		super(str);
	}

	public static void main(String arg[]){
		GUI_Frame1 f = new GUI_Frame1("Marvellous infosystem");
		f.setSize(400,400);

		f.addWindowListener(new ActionListener());
				f.setVisible(true);
	}
}

class ActionListener implements WindowListener{
	
	public void WindowClosing(WindowEvent obj){}{
		System.exit(0);
	}
	

	
	
}

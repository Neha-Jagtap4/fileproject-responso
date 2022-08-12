import java.awt.*;
import java.awt.Event.*;
import java.awt.event.WindowListener;
import java.awt.event.WindowEvent;

class MarvellousWindow extends Frame implements WindowListener{
	public MarvellousWindow(String str,int x,int y){
		super();
		setTitle(str);
		setSize(x,y);
		addWindowListener(this);
		setVisible(true);
	
	}
	public void WindowDeactivated(WindowEvent obj){}  ///errror
	public void WindowActivated(WindowEvent obj){}
	public void WindowDeiconfied(WindowEvent obj){}
	public void WindowIconfied(WindowEvent obj){}
	public void WindowClosed(WindowEvent obj){}
	public void WindowClosing(WindowEvent obj){
	
	System.exit(0);	
}

	public void WindowOpened(WindowEvent obj){}
	
}

class GUI8{
	public static void main(String arg[]){
		MarvellousWindow mobj = new MarvellousWindow("Marvellous",500,500);
	}
}
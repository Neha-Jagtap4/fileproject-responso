import java.awt.*;
//import java.awt.Event.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

class MarvellousWindow extends WindowAdapter{
	Frame fobj;

	public MarvellousWindow(String str,int x,int y){
		fobj = new Frame(str);
		fobj.setSize(x,y);
		fobj.addWindowListener(this);
		fobj.setVisible(true);
	}

	public void WindowClosing(WindowEvent obj){
		System.exit(0);
	}
}

class GUI9{
	public static void main(String arg[]){
		MarvellousWindow mobj = new MarvellousWindow("Marvellous",500,500);
	}
}

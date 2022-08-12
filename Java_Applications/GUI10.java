import java.awt.*;
//import java.awt.Event.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

class MarvellousWindow{
	Frame fobj;
	public MarvellousWindow(String str,int x,int y){
		fobj = new Frame(str);
		fobj.setSize(x,y);
		fobj.setVisible(true);

		fobj.addWindowListener(new WindowAdapter (){
			public void WindowClosing(WindowEvent obj){
				System.exit(0);
			}
		});
	}
}

class GUI10{
	public static void main(String arg[]){
		MarvellousWindow mobj = new MarvellousWindow("Marvellous",500,500);
	}
}
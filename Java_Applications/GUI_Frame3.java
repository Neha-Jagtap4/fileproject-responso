import java.awt.*;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.event.WindowAdapter;

class GUI_Frame3 extends Frame{
	GUI_Frame3(String str){
		super(str);
	}

	public static void main(String arg[]){
		GUI_Frame3 f = new GUI_Frame3("Marvellous Infosystem");
		f.setSize(500,500);
		f.setVisible(true);

		f.addWindowListener(new WindowAdapter(){
			public void WindowClosing(WindowEvent obj){
				System.exit(0);
			}
		});
	}
}
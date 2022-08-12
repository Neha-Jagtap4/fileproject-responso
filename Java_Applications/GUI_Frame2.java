import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

class GUI_Frame2 extends Frame{
	GUI_Frame2(String name){
		super(name);
	}
	public static void main(String arg[]){
		GUI_Frame2 f =new GUI_Frame2("Marvellous Infosystem");
		f.setSize(500,500);
		f.setVisible(true);

		f.addWindowListener(new ActionListener());

	}

}

class ActionListener extends WindowAdapter{
	public void WindowClosing(WindowEvent e){
		System.exit(0);
	}
}
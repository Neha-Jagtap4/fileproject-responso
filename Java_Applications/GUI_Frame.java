import java.awt.*;
import java.awt.event.*;

class GUI_Frame extends Frame{
	GUI_Frame(String name){
		super(name);
	}

	public static void main(String arg[]){
		GUI_Frame f =new GUI_Frame("Marvellous Infosystem");

		f.setSize(500,500);
		f.setVisible(true);
	}
}
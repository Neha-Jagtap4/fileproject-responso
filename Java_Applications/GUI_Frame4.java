import java.awt.*;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowListener;

class GUI_Frame4 extends Frame{
	GUI_Frame4(){
		addWindowListener(new WindowAdapter(){
			public void WindowClosing(WindowEvent obj){
				System.exit(0);
			}
		});
	}

	public void paint(Graphics g){
		this.setBackground(new Color(255,155,155));
	}

	public static void main(String arg[]){
		GUI_Frame4 f = new GUI_Frame4();
		f.setTitle("Marvellous infosystem");
		f.setSize(500,500);
		f.setVisible(true);


	}
}
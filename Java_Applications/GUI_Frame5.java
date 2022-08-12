import java.awt.*;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowListener;

class GUI_Frame5 extends Frame{
	GUI_Frame5(){
		addWindowListener(new WindowAdapter(){
			public void WindowClosing(WindowEvent obj){
				System.exit(0);
			}
		});
	}

	public void paint(Graphics g){
		this.setBackground(new Color(255,255,55));  //frame la color

		Font fn = new Font("Century",Font.BOLD,55);  //font chi size
		g.setFont(fn); //font la set klea aah
		g.drawString("MARVELLOUS INFOSYSTEM",250,250);//frame madhe yett
	}
	public static void main(String arg[]){
		GUI_Frame5 f = new GUI_Frame5();
		f.setTitle("Marvellous infosystem");//heading of frame
		f.setSize(500,500);
		f.setVisible(true);


	}
}
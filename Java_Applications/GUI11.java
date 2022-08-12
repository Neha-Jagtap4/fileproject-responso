import java.awt.*;
//import java.awt.Event.*;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.event.WindowListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;

public class GUI11{
	//charactertics
	Frame mainframe;
	Label HeaderLabel;
	Label StatusLabel;
	Panel ControlLabel;


	//CONSTRUCTOR
	public GUI11(){
	mainframe = new Frame("Marvellous infosystem");
	mainframe.setSize(500,500);
	mainframe.setLayout(new GridLayout(3,1));

	mainframe.addWindowListener(new WindowAdapter(){
		public void WindowClosing(WindowEvent obj){
			System.exit(0);
		}
	});

	HeaderLabel = new Label();
	HeaderLabel.setAlignment(Label.CENTER);

	StatusLabel =  new Label();
	StatusLabel.setAlignment(Label.CENTER);
	StatusLabel.setSize(350,100);

	ControlLabel = new Panel();
	ControlLabel.setLayout(new FlowLayout());

	mainframe.add(HeaderLabel);
	mainframe.add(ControlLabel);
	mainframe.add(StatusLabel);
	mainframe.setVisible(true);
}
	public static void main(String arg[]){
		GUI11 obj = new GUI11();
		obj.Display();
	
}

	private void Display(){
		HeaderLabel.setText("Press select button");
		Button PPA =new Button("PPA");
		Button LB = new Button ("LB");
		Button ANGULAR = new Button("Angular");

		PPA.addActionListener(new myListener());
		LB.addActionListener(new myListener());
		ANGULAR.addActionListener(new myListener());

		ControlLabel.add(PPA);
		ControlLabel.add(LB);
		ControlLabel.add(ANGULAR);

		mainframe.setVisible(true);

	}

	private class myListener implements ActionListener{
		public void actionPerformed(ActionEvent obj){
			String command = obj.getActionCommand();
			if(command.equals("PPA")){
				StatusLabel.setText("Pre-Placement activity");
		}
		else if(command.equals("LB")){
				StatusLabel.setText("Logic buildiing");
		}
		else if(command.equals("ANGULAR")){
			StatusLabel.setText("Web development");
		}
	}
}


}
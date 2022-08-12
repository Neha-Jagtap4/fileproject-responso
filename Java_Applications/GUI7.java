import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

class MarvellousWindow extends Frame implements ActionListener{
	public Button b1;
	public TextField t1;

	public MarvellousWindow(String str,int x,int y){
		super();
		setTitle(str);
		setSize(x,y);
		//setVisible(true);

		b1 = new Button ("Submit");
		t1 = new TextField();

		b1.setBounds(50,50,100,50);
		t1.setBounds(50,100,100,50);

		add(b1);
		add(t1);

		b1.addActionListener(this); //sumbit la iiktoy
		setLayout(null);
		setVisible(true);
	}

	public void actionPerformed(ActionEvent ae){
		t1.setText("PPA");  //PPA submit
	}
}

class GUI7{
	public static void main(String arg[]){
	MarvellousWindow mobj = new MarvellousWindow("Marvellous",500,500);
	}
}
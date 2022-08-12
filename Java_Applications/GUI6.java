import java.awt.*;

class MarvellousWindow extends Frame{
	Button b1;
	TextField t1;

	public MarvellousWindow(String str,int x,int y){
		super();
		setTitle(str);
		setSize(x,y);
		setVisible(true);
		b1 = new Button("Submit");
		t1 = new TextField();
		b1.setBounds(50,50,100,100);
		t1.setBounds(50,100,100,50);

		add(b1);
		add(t1);

		setLayout(null);
		setVisible(true);
	}
}


class GUI6{
	public static void main(String arg[]){
		MarvellousWindow mobj = new MarvellousWindow("Marvellous",500,500);
	}
}
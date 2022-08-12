import java.awt.*;

class MarvellousWindow extends Frame{
	public MarvellousWindow(String str){
		super(str);
	}
}

class GUI1{
	public static void main(String arg[]){
		MarvellousWindow mobj= new MarvellousWindow("Marvellous");

		mobj.setSize(300,300);
		mobj.setVisible(true);
	}
}
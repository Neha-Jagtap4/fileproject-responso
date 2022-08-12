import java.awt.*;

class MarvellousWindow extends Frame{
	public MarvellousWindow(){
		super();
	}
}

class GUI2{
	public static void main(String arg[]){
		MarvellousWindow mobj = new MarvellousWindow();
		mobj.setSize(500,500);
		mobj.setTitle("Marvellous");
		mobj.setVisible(true);
	}
}
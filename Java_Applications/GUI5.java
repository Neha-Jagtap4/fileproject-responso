import java.awt.*;

class MarvellousWindow extends Frame{
	public MarvellousWindow(String str,int x ,int y){
	super();
	setTitle(str);
	setSize(x,y);
	setVisible(true);
}
}

class GUI5{
	public static void main(String arg[]){
		MarvellousWindow mobj1 = new MarvellousWindow("Marvellous",500,500);
	// /	MarvellousWindow mobj2 = new MarvellousWindow("Marvellous1",200,200);

	}
}
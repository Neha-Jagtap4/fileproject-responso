import java.awt.*;

class Demo{
	Demo(){
		Frame f = new Frame("Marvellous infosystem");
		Label SName = new Label("Student name");
		SName.setBounds(20,50,120,20);

		Label BName = new Label("Batch name");
		BName.setBounds(20,80,120,20);

		TextField SNameTF = new TextField();
		SNameTF.setBounds(140,50,100,20);

		TextField BNameTF = new TextField();
		BNameTF.setBounds(140,80,100,20);

		Button Submit = new Button(SUBMIT);
		Submit.setBounds(200,100,100,30);

		Button Reset = new Button(RESET);
		Reset.setBounds(120,160,100,30);

		f.add(SName);
		f.add(BName);
		f.add(SNameTF);
		f.add(BNameTF);
		f.add(Submit);
		f.add(Reset);

		f.setSize(400,400);
		f.setLayout(null);
		f.setVisible(true);

	}

}

class AWT{
	public static void main(String arg[]){
		Demo d = new Demo();
	}
}
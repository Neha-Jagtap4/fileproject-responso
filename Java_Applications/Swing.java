import javax.swing.*;

class Demo{
	Demo(){
		JFrame f = new JFrame("Marvellous infosystem");
		JLabel SName = new JLabel("Student name");
		SName.setBounds(20,50,120,20);

		JLabel BName = new JLabel("Batch name");
		BName.setBounds(20,80,120,20);

		JTextField SNameTF = new JTextField();
		SNameTF.setBounds(140,50,100,20);

		JTextField BNameTF = new JTextField();
		BNameTF.setBounds(140,80,100,20);

		JButton Submit = new JButton("SUBMIT");
		Submit.setBounds(200,100,100,30);

		JButton Reset = new JButton("RESET");
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

class Swing{
	public static void main(String arg[]){
		Demo d = new Demo();
	}
}
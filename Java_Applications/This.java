class Demo{
	public int x,y;
	public Demo(){
		System.out.println("Default constructor");
		this.x=0;
		this.y=0;
	}

	public Demo(int i,int j){
		System.out.println("Paramertized constructor");
		this.x=i;
		this.y=j;
	}

	public void fun(){
		System.out.println("Inside function");
		 
	}

	public void gun(){
		System.out.println("Inside function1");
		
	}


}

class This{
	public static void main(String str[]){
		Demo dobj = new Demo(10,20);
		dobj.fun();
		dobj.gun();

	}
}
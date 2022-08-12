class Base{
	int x,y;
	public Base(){
		System.out.println("Default constructor");
		this.x=0;
		this.y=0;
	}

	public Base(int i,int j){
		System.out.println("Base Paramertized Constructor");
		this.x=i;
		this.y=j;
	}

	public void fun(){
		System.out.println("Inside function Base");
	}

}

class Derived extends Base{
	public int a,b;
	public Derived(){
		System.out.println("Default Constructor");
		this.a=0;
		this.b=0;
	}
	public Derived(int iNo1,int iNo2){
		System.out.println("Derived paramertized Constructor");
		this.a=iNo1;
		this.b=iNo2;

	}

	public void fun(){
		System.out.println("Inside function derived");

	}
}

class Inheritance{
	public static void main(String arg[]){
		Base bobj = new Base();
		bobj.fun();

		Derived dobj= new Derived();
		dobj.fun();
	}
}
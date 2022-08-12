//Write a program of addition by overloading the function.
class Demo{
	public int Add(int iNo1,int iNo2){
		return iNo1+iNo2;
	}

	public float Adds(float iNo1,float iNo2){
		return iNo1+iNo2;
	}

	public int Add2(int iNo1,int iNo2,int iNo3){
		return iNo1+iNo2+iNo3;
	}

	public void fun(int i,float j){
		System.out.println("Inside first function");
	}

	public void fun(float j,int i){
		System.out.println("Inside second function");
	}


}

class Overloading{
	public static void main(String str[]){
		Demo dobj=new Demo();
		int  iRet=0;
		float fRet=0.0f;

		iRet=dobj.Add(10,20);
		System.out.println("Addition is :" +iRet);

		fRet=dobj.Adds(10,11);
		System.out.println("Addition is :" +fRet);

		iRet=dobj.Add2(10,20,30);
		System.out.println("Addition is :" +iRet);

		dobj.fun(10,11.3f);
		dobj.fun(45.2f,15);

	}
}
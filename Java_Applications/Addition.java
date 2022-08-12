class Arithmetic{
	public int iNo1,iNo2;

	public Arithmetic(){
		this.iNo1=0;
		this.iNo2=0;
	}
	public Arithmetic(int i,int j){
		this.iNo1=i;
		this.iNo2=j;
	}

	public int Addition(){
		int iAns=0;
		iAns=this.iNo1+this.iNo2;
		return iAns;
	}

	public int Subtraction(){
		int iAns=0;
		iAns=this.iNo1-this.iNo2;
		return iAns;
	}
};

class Addition{
	public static void main(String str[]){
		Arithmetic obj2=new Arithmetic(10,20);
		int iRet=0;
		
		iRet=obj2.Addition();
		System.out.println("Adition is :" +iRet);

		iRet=obj2.Subtraction();
		System.out.println("Subtraction is :" +iRet);
	}
}
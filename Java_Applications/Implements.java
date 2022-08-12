import java.util.*;

class Implements{

	public static void main(String arg[]){
		float value = 0.0f;
		Scanner sobj = new Scanner(System.in);
		System.out.println("Enter Radius");
	 	value = sobj.nextFloat();   //input ghetoy

		Marvellous mobj = new Marvellous();
		float fret= 0.0f;
		fret = mobj.circumference(value);//scanf

		System.out.println("Area is = " +fRet);
		System.out.println("circumference is =" +fret);
		System.out.println("Value of PI :" +circle.PI);
	}
}

interface circle{
	float PI = 3.14f;
	float area(float radius);
	float circumference(float Radius);
}

class Marvellous implements circle{
	public float area(float radius){
		float ret = 0.0f;
		ret=PI*radius*radius;
		return ret;
}

	public float circumference(float Radius){
		float ret = 0.0f;
		ret = 2* PI*Radius;//2*3.14*radius
		return ret;
}


}
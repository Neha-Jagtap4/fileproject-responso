import java.util.*;


class Sort{
	public static void main(String arg[]){
		Scanner sobj = new Scanner(System.in);
		System.out.println("Enter the size of array");
		int size = sobj.nextInt();

		int Arr[] = new int[size];
		System.out.println("Enter the elements");
		for(int i=0; i<Arr.length; i++){
			Arr[i] = sobj.nextInt();
		}

		System.out.println("Entered data is:");
		for(int i=0; i<Arr.length ; i++){
			System.out.println(Arr[i]);
		}
	}
}
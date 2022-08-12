import java.util.*;

class JavaAdv{
	public static void main(String arg[]){
		int i=0,j=0;
		Scanner sobj = new Scanner(System.in);

		System.out.println("Enter number of rows");
		int rows = sobj.nextInt();

		System.out.println("Enter the columns");
		int column = sobj.nextInt();

		int Arr[][] = new int[rows][column];

		System.out.println("Enter the elemnts");

		for(i=0; i<Arr.length; i++){
			for(j=0; j<Arr[i].length; j++){
				Arr[i][j] = sobj.nextInt();
			}
		}

		System.out.println("Enter elements are");
		for(i=0; i<Arr.length; i++){
			for(j=0; j<Arr[i].length; j++){
				System.out.println(Arr[i][j]+"\t");
			}
			System.out.println();
		}
	}
}
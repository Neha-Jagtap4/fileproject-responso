
class String{
	public static void main(String arg[]){
String sobj1 = "Hello";
String sobj2 = "Hello";
String sobj3 = new String("Hello");

if(sobj1 == sobj3){
	System.out.println("equal");
}
else{
	System.out.println("not equal");
}
}
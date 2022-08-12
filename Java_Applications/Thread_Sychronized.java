import java.util.*;

class Demo extends Thread{
	synchronized void Display(){
		for(int i=0;i<=100; i++){
			System.out.println("Thread" +this.getName()+ ":" +i);  //mhanun student ek-ek yett aahet
		}
	}
	public void gun(){
		Display();  //teacher aah
	}
}

class Thread_Sychronized{
	public static void main(String arg[]){
		Demo d1 = new Demo();
		Demo d2 = new Demo();

		Thread t1 = new Thread(d1,"first");
		Thread t2 = new Thread(d2,"Second");

		System.out.println("Inside main");
		t1.start();
		t2.start();
	}
}
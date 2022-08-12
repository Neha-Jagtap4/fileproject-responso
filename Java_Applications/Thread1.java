import java.util.*;

class Demo extends Thread{
	public void fun(){
		for(int i=1; i<10; i++){
			try{
				Thread.sleep(1000);
				System.out.println("Inside :" +Thread.currentThread().getName()+ ":" +i);

			}

			catch(Exception iobj){
				//System.out.println("Exception occured" +iobj);
			}
		}
	}
}

class Thread1
{
	public static void main(String arg[])throws Exception{
		System.out.println("Thread name is :" +Thread.currentThread().getName());
		Demo obj1 = new Demo();
		Demo obj2= new Demo();

		obj1.setName("first");
		obj2.setName("Second");

		obj1.start();
		obj2.start();

		obj1.join();
		obj2.join();

		System.out.println("End of main");

	}
}
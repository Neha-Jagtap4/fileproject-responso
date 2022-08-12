import java.lang.*;

class Demo extends Thread{
	public void run(){
		System.out.println("Threradname" +Thread.currentThread().getName());
		System.out.println("Threradname " +Thread.currentThread().getName());
	}
}

class Thread_Priority{
	public static void main(String arg[]){
		Demo d1 = new Demo();

		Thread t1 = new Thread(d1,"first");
		Thread t2 = new Thread(d1,"Second");

		t1.setpriority(3);
		t2.setpriority(Thread_NORM_PRIORITY);

		t1.start();
		t2.start();
	}
}
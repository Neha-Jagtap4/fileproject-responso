import MarvellousPPA.PackageX;


class Package{
	public static void main(String arg[]){
		PackageX mobj = new PackageX();
		int ret  = mobj.Addition(10,20);
		System.out.println("Addition is: "+ret);
	}
}
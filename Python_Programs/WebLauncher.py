from sys import *
import webbrowser
import re
#import urllib2
import urllib.request as urllib2

def is_connected():
	try:
		#urllib2.urlopen("http://216.58/192.142",timeout=1)
		urllib2.urlopen("https://www.facebook.com/",timeout=1)
		urllib2.urlopen("https://www.instagram.com/",timeout=1)

		return True
	except urllib2.URLError as err:
		return False

def Find(string):
	url=re.findall("https[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",string)
	return url

def WebLauncher(path):
	with open(path) as fp:
		for line in fp:
			print(line)
			url = Find(line)
			print(url)
			for str in url:
				webbrowser.open(str,new=2)

def main():
	print("------Application-------")
	print("Application name : "+argv[0])

	if(len(argv)!=2):
		print("Error : Invalid number of arguments")
		exit()

	if(argv[1]=="-h") or (argv[1]=="-H"):
		print("This script is used to open URL which are written in one file")
		exit()

	if(argv[1]=="-u") or (argv[1]=="-U"):
		print("Usage : ApplicationName Name_of_file")
		exit()

	try:
		connected = is_connected()

		if connected:
			WebLauncher(argv[1])
		else:
			print("Unable to connect internet")
		

	except ValueError:
		print("Error : Invalid datatype of input")

	except Exception as E:
		print("Error : Invalid input",E)


if __name__ == "__main__":
	main()
"""
A Python script to download and play a media file by FTP. Uses ftplib, the ftp protocol handler
which uses sockets. Ftp runs on 2 sockets (one for data, one for control--on ports 20 and 21) and imposes message
text formats, but Python's ftplib module hides most of this protocl's details. Change for your site/file
"""

import os, sys
from getpass import getpass  # hidden password input
from ftplib import FTP  # socket-based FTP tools

nonpassive = False  # force active mode FTP for server?
filename = 'monkeys.jpg'  # file to be downloaded
dirname = '.'  # remote directory to fetch form
sitename = 'ftp.rmi.net'  # FTP site to contact
userinfo = ('lutz', getpass('Pswd?'))  # use () for anonymous
if len(sys.argv) > 1: filename = sys.argv[1]  # filename on command line?

print('Connecting...')  # connecting to FTP site
connection = FTP(sitename)
connection.login(*userinfo) # default is anon
connection.cwd(dirname)     # xfer 1k at a time to localfile
if nonpassive:
    connection.set_pasv(False)      # force active FTP if server requires

print('Downloading...')
localfile = open(filename, 'wb')        # localfile to store download
connection.retrbinary('RETR ' + filename, localfile.write, 1024)
connection.quit()
localfile.close()

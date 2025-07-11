# === Upload the generated image to your web server ===
from ftplib import FTP

# FTP credentials (replace with your real details)
FTP_HOST = "grhost.info"             # e.g. ftp.example.com
FTP_USER = "parognosis"               # e.g. parognosis_user
FTP_PASS = "#6934225193#"               # e.g. secretpassword
FTP_FOLDER = "meteograms"        # folder path on your server

print("🌐 Connecting to FTP server...")
ftp = FTP(FTP_HOST)
ftp.login(FTP_USER, FTP_PASS)

# Change to the target directory
ftp.cwd(FTP_FOLDER)

# Open the file and upload it
with open("athens_meteogram.png", "rb") as file:
    ftp.storbinary("STOR athens_meteogram.png", file)

ftp.quit()
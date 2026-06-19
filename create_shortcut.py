import os
import win32com.client

desktop = r"C:\Users\vinay\Desktop"
path = os.path.join(desktop, "AML Dashboard.lnk")
target = r"c:\project 3\run_aml_dashboard.bat"
icon = r"c:\project 3\run_aml_dashboard.bat"

shell = win32com.client.Dispatch("WScript.Shell")
shortcut = shell.CreateShortCut(path)
shortcut.Targetpath = target
shortcut.WorkingDirectory = r"c:\project 3"
shortcut.WindowStyle = 1 # Normal window
shortcut.save()
print("Shortcut created at:", path)
